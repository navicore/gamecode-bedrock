use anyhow::Result;
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    types::{
        ContentBlock as BedrockContentBlock, ConversationRole, InferenceConfiguration,
        Message as BedrockMessage, Tool as BedrockTool, ToolConfiguration, ToolInputSchema,
        ToolResultBlock, ToolResultContentBlock, ToolUseBlock,
    },
    Client,
};
use aws_smithy_types::{Document, Number};
use gamecode_backend::{
    BackendError, BackendResult, BackendStatus, ChatRequest, ChatResponse, ChatStream,
    ContentBlock, InferenceConfig, LLMBackend, Message, MessageRole, RetryConfig, Tool, ToolCall,
    Usage,
};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use tracing::debug;
use uuid::Uuid;

/// AWS Bedrock implementation of the LLM backend
pub struct BedrockBackend {
    client: Client,
    #[allow(dead_code)]
    region: String,
    default_model: String,
}

impl BedrockBackend {
    /// Create a new Bedrock backend with default AWS configuration
    pub async fn new() -> Result<Self> {
        Self::new_with_region("us-west-2").await
    }

    /// Create a new Bedrock backend with specific region
    pub async fn new_with_region(region: &str) -> Result<Self> {
        let config = aws_config::defaults(BehaviorVersion::latest())
            .region(aws_config::Region::new(region.to_string()))
            .load()
            .await;

        let client = Client::new(&config);

        Ok(Self {
            client,
            region: region.to_string(),
            default_model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0".to_string(),
            //default_model: "us.anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
            //default_model: "us.anthropic.claude-opus-4-20250514-v1:0".to_string(),
        })
    }

    /// Create a new Bedrock backend with custom AWS config
    pub fn new_with_config(config: &aws_config::SdkConfig) -> Self {
        let client = Client::new(config);
        let region = config
            .region()
            .map(|r| r.as_ref().to_string())
            .unwrap_or_else(|| "us-west-2".to_string());

        Self {
            client,
            region,
            default_model: "us.anthropic.claude-3-7-sonnet-20250219-v1:0".to_string(),
            //default_model: "us.anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
            //default_model: "us.anthropic.claude-opus-4-20250514-v1:0".to_string(),
        }
    }

    /// Set the default model for this backend
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    async fn raw_chat(&self, request: ChatRequest) -> BackendResult<ChatResponse> {
        let model_id = request.model.as_ref().unwrap_or(&self.default_model);

        // Log the actual model being used for this request
        debug!("Bedrock API request using model: {}", model_id);

        // Convert request to Bedrock format
        let bedrock_messages = self.convert_messages(&request.messages)?;
        let bedrock_tools = self.convert_tools(&request.tools.unwrap_or_default())?;
        let inference_config = self.convert_inference_config(&request.inference_config);
        
        // Log the converted messages
        debug!("=== BEDROCK REQUEST DETAILS ===");
        debug!("Model: {}", model_id);
        debug!("Number of messages: {}", bedrock_messages.len());
        for (i, msg) in bedrock_messages.iter().enumerate() {
            debug!("Message {}: role={:?}", i, msg.role());
            for (j, block) in msg.content().iter().enumerate() {
                match block {
                    BedrockContentBlock::Text(text) => {
                        debug!("  Content block {}: Text ({})", j, text);
                    }
                    BedrockContentBlock::ToolUse(tool_use) => {
                        debug!("  Content block {}: ToolUse (name={}, id={})", 
                            j, tool_use.name(), tool_use.tool_use_id());
                        debug!("    Input: {}", serde_json::to_string_pretty(&document_to_json_value(tool_use.input())).unwrap_or_default());
                    }
                    BedrockContentBlock::ToolResult(tool_result) => {
                        debug!("  Content block {}: ToolResult (id={})", j, tool_result.tool_use_id());
                    }
                    _ => {
                        debug!("  Content block {}: Other", j);
                    }
                }
            }
        }
        
        debug!("Number of tools: {}", bedrock_tools.len());
        for (i, tool) in bedrock_tools.iter().enumerate() {
            match tool {
                BedrockTool::ToolSpec(spec) => {
                    debug!("Tool {}: {} - {:?}", i, spec.name(), spec.description());
                    if let Some(ToolInputSchema::Json(schema)) = spec.input_schema() {
                        let schema_json = document_to_json_value(schema);
                        debug!("  Schema: {}", serde_json::to_string_pretty(&schema_json).unwrap_or_default());
                    }
                }
                _ => {
                    debug!("Tool {}: Unknown tool type", i);
                }
            }
        }
        
        debug!("Inference config: temperature={:?}, top_p={:?}, max_tokens={:?}", 
            inference_config.temperature(), 
            inference_config.top_p(), 
            inference_config.max_tokens());
        debug!("=== END REQUEST DETAILS ===");

        // Build request
        let mut request_builder = self
            .client
            .converse()
            .model_id(model_id)
            .inference_config(inference_config);

        // Add messages
        for message in bedrock_messages {
            request_builder = request_builder.messages(message);
        }

        // Add tools if present
        if !bedrock_tools.is_empty() {
            let mut tool_config_builder = ToolConfiguration::builder();
            for tool in bedrock_tools {
                tool_config_builder = tool_config_builder.tools(tool);
            }
            let tool_config =
                tool_config_builder
                    .build()
                    .map_err(|e| BackendError::InternalError {
                        message: format!("Failed to build tool configuration: {}", e),
                    })?;
            request_builder = request_builder.tool_config(tool_config);
        }

        // Send request
        let response = match request_builder.send().await {
            Ok(resp) => resp,
            Err(e) => {
                debug!("=== BEDROCK ERROR DETAILS ===");
                debug!("Full error: {:?}", e);
                
                // Try to extract service error details
                use aws_sdk_bedrockruntime::operation::converse::ConverseError;
                if let aws_sdk_bedrockruntime::error::SdkError::ServiceError(service_err) = &e {
                    debug!("Service error: {:?}", service_err);
                    
                    match service_err.err() {
                        ConverseError::ThrottlingException(ex) => {
                            debug!("  ThrottlingException: {:?}", ex.message());
                        }
                        ConverseError::ValidationException(ex) => {
                            debug!("  ValidationException: {:?}", ex.message());
                        }
                        _ => {
                            debug!("  Other error type");
                        }
                    }
                }
                debug!("=== END ERROR DETAILS ===");
                return Err(self.convert_error(e));
            }
        };

        // Log response details
        debug!("=== BEDROCK RESPONSE DETAILS ===");
        if let Some(usage) = response.usage() {
            debug!("Token usage: input={}, output={}, total={}", 
                usage.input_tokens(), usage.output_tokens(), usage.total_tokens());
        }
        debug!("=== END RESPONSE DETAILS ===");

        // Convert response
        self.convert_response(response, model_id.clone(), request.session_id)
    }

    fn convert_messages(&self, messages: &[Message]) -> BackendResult<Vec<BedrockMessage>> {
        let mut bedrock_messages = Vec::new();

        for message in messages {
            let role = match message.role {
                MessageRole::System => ConversationRole::User, // Bedrock treats system as user
                MessageRole::User => ConversationRole::User,
                MessageRole::Assistant => ConversationRole::Assistant,
            };

            let mut content_blocks = Vec::new();
            for content in &message.content {
                match content {
                    ContentBlock::Text(text) => {
                        content_blocks.push(BedrockContentBlock::Text(text.clone()));
                    }
                    ContentBlock::ToolCall(tool_call) => {
                        let tool_use = ToolUseBlock::builder()
                            .tool_use_id(&tool_call.id)
                            .name(&tool_call.name)
                            .input(json_value_to_document(&tool_call.input))
                            .build()
                            .map_err(|e| BackendError::InternalError {
                                message: format!("Failed to build tool use block: {}", e),
                            })?;
                        content_blocks.push(BedrockContentBlock::ToolUse(tool_use));
                    }
                    ContentBlock::ToolResult {
                        tool_call_id,
                        result,
                    } => {
                        let tool_result = ToolResultBlock::builder()
                            .tool_use_id(tool_call_id)
                            .content(ToolResultContentBlock::Text(result.clone()))
                            .build()
                            .map_err(|e| BackendError::InternalError {
                                message: format!("Failed to build tool result block: {}", e),
                            })?;
                        content_blocks.push(BedrockContentBlock::ToolResult(tool_result));
                    }
                }
            }

            let bedrock_message = BedrockMessage::builder()
                .role(role)
                .set_content(Some(content_blocks))
                .build()
                .map_err(|e| BackendError::InternalError {
                    message: format!("Failed to build message: {}", e),
                })?;

            bedrock_messages.push(bedrock_message);
        }

        Ok(bedrock_messages)
    }

    fn convert_tools(&self, tools: &[Tool]) -> BackendResult<Vec<BedrockTool>> {
        let mut bedrock_tools = Vec::new();

        for tool in tools {
            let tool_spec = aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                .name(&tool.name)
                .description(&tool.description)
                .input_schema(ToolInputSchema::Json(json_value_to_document(
                    &tool.input_schema,
                )))
                .build()
                .map_err(|e| BackendError::InternalError {
                    message: format!("Failed to build tool specification: {}", e),
                })?;

            bedrock_tools.push(BedrockTool::ToolSpec(tool_spec));
        }

        Ok(bedrock_tools)
    }

    fn convert_inference_config(&self, config: &Option<InferenceConfig>) -> InferenceConfiguration {
        let default_config = InferenceConfig::default();
        let config = config.as_ref().unwrap_or(&default_config);

        InferenceConfiguration::builder()
            .set_temperature(config.temperature)
            .set_top_p(config.top_p)
            .set_max_tokens(config.max_tokens.map(|t| t as i32))
            .build()
    }

    fn convert_response(
        &self,
        response: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
        model: String,
        session_id: Option<Uuid>,
    ) -> BackendResult<ChatResponse> {
        let output = response
            .output()
            .ok_or_else(|| BackendError::ProviderError {
                message: "No output in response".to_string(),
            })?;

        let bedrock_message = output
            .as_message()
            .map_err(|_| BackendError::ProviderError {
                message: "Response output is not a message".to_string(),
            })?;

        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        for content in bedrock_message.content() {
            match content {
                BedrockContentBlock::Text(text) => {
                    text_content.push_str(text);
                }
                BedrockContentBlock::ToolUse(tool_use) => {
                    let tool_call = ToolCall {
                        id: tool_use.tool_use_id().to_string(),
                        name: tool_use.name().to_string(),
                        input: document_to_json_value(tool_use.input()),
                    };
                    tool_calls.push(tool_call);
                }
                _ => {} // Ignore other content types for now
            }
        }

        let message = if !text_content.is_empty() && !tool_calls.is_empty() {
            Message::with_tool_calls(MessageRole::Assistant, text_content, tool_calls.clone())
        } else if !text_content.is_empty() {
            Message::text(MessageRole::Assistant, text_content)
        } else {
            Message {
                role: MessageRole::Assistant,
                content: tool_calls
                    .iter()
                    .map(|tc| ContentBlock::ToolCall(tc.clone()))
                    .collect(),
            }
        };

        let usage = response.usage().map(|u| Usage {
            input_tokens: u.input_tokens() as u32,
            output_tokens: u.output_tokens() as u32,
            total_tokens: u.total_tokens() as u32,
        });

        Ok(ChatResponse {
            message,
            tool_calls,
            usage,
            model,
            session_id,
        })
    }

    fn convert_error(
        &self,
        error: aws_sdk_bedrockruntime::error::SdkError<
            aws_sdk_bedrockruntime::operation::converse::ConverseError,
        >,
    ) -> BackendError {
        use aws_sdk_bedrockruntime::operation::converse::ConverseError;
        
        // First check for service errors with proper types
        if let aws_sdk_bedrockruntime::error::SdkError::ServiceError(service_err) = &error {
            debug!("Bedrock service error: {:?}", service_err.err());
            
            match service_err.err() {
                ConverseError::ThrottlingException(_e) => {
                    debug!("Detected ThrottlingException");
                    return BackendError::RateLimited;
                }
                ConverseError::ValidationException(e) => {
                    debug!("Detected ValidationException: {:?}", e.message());
                    return BackendError::ValidationError {
                        message: e.message().unwrap_or("Validation error").to_string(),
                    };
                }
                ConverseError::ResourceNotFoundException(e) => {
                    debug!("Detected ResourceNotFoundException: {:?}", e.message());
                    return BackendError::ValidationError {
                        message: format!("Resource not found: {}", e.message().unwrap_or("unknown")),
                    };
                }
                _ => {
                    debug!("Unknown service error type");
                }
            }
        }
        
        // Fallback to string matching for other error types
        let error_str = format!("{:?}", error);
        debug!("Bedrock API error (full): {}", error_str);

        // Default to provider error
        BackendError::ProviderError {
            message: extract_error_summary(&error_str),
        }
    }
}

#[async_trait]
impl LLMBackend for BedrockBackend {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        self.chat_with_retry(request, RetryConfig::default()).await
    }

    async fn chat_stream(&self, _request: ChatRequest) -> Result<ChatStream> {
        // TODO: Implement streaming support
        todo!("Streaming support not yet implemented")
    }

    async fn chat_with_retry(
        &self,
        request: ChatRequest,
        retry_config: RetryConfig,
    ) -> Result<ChatResponse> {
        retry_with_backoff::<_, _, ChatResponse>(
            || self.raw_chat(request.clone()),
            retry_config.max_retries,
            retry_config.initial_delay,
            request.status_callback.as_ref(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Chat request failed: {}", e))
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supported_models(&self) -> Vec<String> {
        vec![
            "us.anthropic.claude-opus-4-20250514-v1:0".to_string(),
            "us.anthropic.claude-sonnet-4-20250514-v1:0".to_string(),
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0".to_string(),
            "anthropic.claude-3-5-sonnet-20240620-v1:0".to_string(),
            "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
            "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
            "anthropic.claude-3-haiku-20240307-v1:0".to_string(),
        ]
    }

    fn default_model(&self) -> String {
        self.default_model.clone()
    }
}

// Helper function to perform exponential backoff retry for AWS Bedrock calls
async fn retry_with_backoff<F, Fut, T>(
    operation: F,
    max_retries: usize,
    initial_delay: Duration,
    status_callback: Option<&gamecode_backend::StatusCallback>,
) -> BackendResult<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = BackendResult<T>>,
{
    let mut delay = initial_delay;
    let mut last_error = None;

    for attempt in 0..=max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                let error_str = format!("{:?}", error);
                let concise_error = extract_error_summary(&error_str);

                if attempt == max_retries {
                    last_error = Some(error);
                    break;
                }

                // Don't retry validation errors
                if !error.is_retryable() {
                    if let Some(callback) = status_callback {
                        callback(BackendStatus::NonRetryableError {
                            message: concise_error,
                        });
                    }
                    last_error = Some(error);
                    break;
                }

                let is_throttling = matches!(error, BackendError::RateLimited);
                if let Some(callback) = status_callback {
                    if is_throttling {
                        callback(BackendStatus::RateLimited {
                            attempt: attempt + 1,
                            max_attempts: max_retries + 1,
                            delay_ms: delay.as_millis() as u64,
                        });
                    } else {
                        callback(BackendStatus::RetryAttempt {
                            attempt: attempt + 1,
                            max_attempts: max_retries + 1,
                            delay_ms: delay.as_millis() as u64,
                            reason: concise_error,
                        });
                    }
                }

                debug!(
                    "Attempt {}/{} failed, retrying after {}ms: {}",
                    attempt + 1,
                    max_retries + 1,
                    delay.as_millis(),
                    error_str
                );

                sleep(delay).await;

                // Exponential backoff with cap
                delay = std::cmp::min(delay * 2, Duration::from_secs(20));
                last_error = Some(error);
            }
        }
    }

    // If we get here, all retries failed
    if let Some(error) = last_error {
        return Err(error);
    }

    unreachable!("Should not reach here")
}

// Helper function to extract concise error information
fn extract_error_summary(error_str: &str) -> String {
    // Look for common AWS error patterns
    if let Some(start) = error_str.find("ServiceError") {
        if let Some(end) = error_str[start..].find(", ") {
            let service_error = &error_str[start..start + end];
            if let Some(err_start) = service_error.find("err: ") {
                if let Some(err_end) = service_error[err_start..].find("(") {
                    let error_type = &service_error[err_start + 5..err_start + err_end];
                    return format!("ServiceError source: {}", error_type);
                }
            }
            return service_error.to_string();
        }
    }

    // Look for specific exceptions
    for exception in &[
        "ThrottlingException",
        "ValidationException",
        "UnauthorizedException",
    ] {
        if error_str.contains(exception) {
            return exception.to_string();
        }
    }

    // Generic patterns
    if error_str.contains("Too many requests") {
        return "Rate limit exceeded".to_string();
    }

    // Fallback to first part
    if let Some(newline_pos) = error_str.find('\n') {
        error_str[..newline_pos].to_string()
    } else if error_str.len() > 100 {
        format!("{}...", &error_str[..100])
    } else {
        error_str.to_string()
    }
}

// Helper function to convert JSON Value to Document
fn json_value_to_document(value: &Value) -> Document {
    match value {
        Value::String(s) => Document::String(s.clone()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Document::Number(Number::PosInt(i as u64))
            } else if let Some(f) = n.as_f64() {
                Document::Number(Number::Float(f))
            } else {
                Document::String(n.to_string())
            }
        }
        Value::Bool(b) => Document::Bool(*b),
        Value::Null => Document::Null,
        Value::Array(arr) => {
            let doc_vec: Vec<Document> = arr.iter().map(json_value_to_document).collect();
            Document::Array(doc_vec)
        }
        Value::Object(obj) => {
            let mut doc_map = HashMap::new();
            for (k, v) in obj {
                doc_map.insert(k.clone(), json_value_to_document(v));
            }
            Document::Object(doc_map)
        }
    }
}

// Helper function to convert Document to JSON Value
fn document_to_json_value(doc: &Document) -> Value {
    match doc {
        Document::String(s) => Value::String(s.clone()),
        Document::Number(n) => match n {
            Number::PosInt(i) => Value::Number(serde_json::Number::from(*i)),
            Number::NegInt(i) => Value::Number(serde_json::Number::from(*i)),
            Number::Float(f) => Value::Number(
                serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0)),
            ),
        },
        Document::Bool(b) => Value::Bool(*b),
        Document::Null => Value::Null,
        Document::Array(arr) => Value::Array(arr.iter().map(document_to_json_value).collect()),
        Document::Object(obj) => {
            let mut map = serde_json::Map::new();
            for (k, v) in obj {
                map.insert(k.clone(), document_to_json_value(v));
            }
            Value::Object(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamecode_backend::{Message, MessageRole};

    #[tokio::test]
    async fn test_backend_creation() {
        // Note: This test requires AWS credentials to be configured
        // In a real test environment, you'd use mocked AWS clients
        if std::env::var("AWS_ACCESS_KEY_ID").is_ok() {
            let backend = BedrockBackend::new().await;
            assert!(backend.is_ok());
        }
    }

    #[test]
    fn test_supported_models() {
        let config = aws_config::SdkConfig::builder()
            .behavior_version(BehaviorVersion::latest())
            .build();
        let backend = BedrockBackend::new_with_config(&config);
        let models = backend.supported_models();
        assert!(!models.is_empty());
        //assert!(models.contains(&"us.anthropic.claude-3-7-sonnet-20250219-v1:0".to_string()));
        assert!(models.contains(&"us.anthropic.claude-opus-4-20250514-v1:0".to_string()));
    }

    #[test]
    fn test_message_conversion() {
        let config = aws_config::SdkConfig::builder()
            .behavior_version(BehaviorVersion::latest())
            .build();
        let backend = BedrockBackend::new_with_config(&config);

        let messages = vec![Message::text(MessageRole::User, "Hello")];
        let result = backend.convert_messages(&messages);
        assert!(result.is_ok());

        let bedrock_messages = result.unwrap();
        assert_eq!(bedrock_messages.len(), 1);
    }
}
