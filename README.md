# gamecode-bedrock

AWS Bedrock implementation for the gamecode-backend trait with production-ready retry logic and error handling.

## Overview

This crate provides a robust AWS Bedrock implementation of the `LLMBackend` trait from `gamecode-backend`. It includes sophisticated retry logic with exponential backoff, comprehensive error handling, and support for Claude 3.7, 3.5, and 3.0 models.

## Features

- **Production-Ready Retry Logic**: Exponential backoff with configurable parameters
- **Comprehensive Error Handling**: Proper categorization of AWS errors with retry guidance
- **Tool Calling Support**: Full support for Claude's function/tool calling capabilities
- **Multiple Claude Models**: Support for Claude 3.7 Sonnet, Claude 3.5 Sonnet/Haiku, and Claude 3.0 models
- **Flexible Configuration**: Support for custom AWS configs and regions

## Usage

### Basic Usage

```rust
use gamecode_bedrock::BedrockBackend;
use gamecode_backend::{ChatRequest, Message, MessageRole};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create backend (uses default AWS credential chain)
    let backend = BedrockBackend::new().await?;
    
    // Create a chat request
    let request = ChatRequest {
        messages: vec![Message::text(MessageRole::User, "Hello, Claude!")],
        model: None, // Uses default model
        tools: None,
        inference_config: None,
        session_id: None,
    };
    
    // Send request
    let response = backend.chat(request).await?;
    println!("Response: {:?}", response.message);
    
    Ok(())
}
```

### Custom Configuration

```rust
use gamecode_bedrock::BedrockBackend;
use gamecode_backend::RetryConfig;
use std::time::Duration;

// Custom region
let backend = BedrockBackend::new_with_region("us-east-1").await?;

// Custom model
let backend = BedrockBackend::new()
    .await?
    .with_default_model("anthropic.claude-3-5-sonnet-20240620-v1:0");

// Custom retry configuration
let retry_config = RetryConfig {
    max_retries: 5,
    initial_delay: Duration::from_millis(1000),
    backoff_strategy: BackoffStrategy::Exponential { multiplier: 2 },
    verbose: true,
};

let response = backend.chat_with_retry(request, retry_config).await?;
```

### With Tools

```rust
use gamecode_backend::{Tool, InferenceConfig};
use serde_json::json;

let tools = vec![
    Tool {
        name: "get_weather".to_string(),
        description: "Get current weather for a location".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }),
    }
];

let request = ChatRequest {
    messages: vec![Message::text(MessageRole::User, "What's the weather in Tokyo?")],
    model: None,
    tools: Some(tools),
    inference_config: Some(InferenceConfig {
        temperature: Some(0.3),
        top_p: Some(0.9),
        max_tokens: Some(1000),
    }),
    session_id: None,
};

let response = backend.chat(request).await?;
```

## Supported Models

- `us.anthropic.claude-3-7-sonnet-20250219-v1:0` (default)
- `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `anthropic.claude-3-5-haiku-20241022-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`

## Error Handling

The backend automatically categorizes AWS errors and provides appropriate retry behavior:

- **Rate Limiting**: Automatic retry with exponential backoff
- **Validation Errors**: No retry (client error)
- **Authentication Errors**: No retry (configuration issue)
- **Network Errors**: Automatic retry
- **Provider Errors**: Automatic retry

## Requirements

- AWS credentials configured (via environment variables, IAM role, or AWS config files)
- Access to AWS Bedrock in your configured region
- Appropriate IAM permissions for Bedrock model access

## License

MIT