use async_trait::async_trait;
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{to_string, to_string_pretty, Map, Value};
use std::future::Future;
use std::{collections::HashMap, fmt::Error};

pub const OPENAI_API_KEY_NAME: &'static str = "OPENAI_API_KEY";
pub const OPENAI_API_BASE: &'static str = "api.openai.com/v1";
pub const OPENAI_API_CHAT_ENDPOINT: &'static str = "/chat/completions";

#[async_trait]
pub trait Model {
    fn classify(
        &self,
        instruction: String,
        text: String,
        choices: Vec<String>,
    ) -> impl Future<Output = Result<usize, ClassifyError>> + Send;

    fn binary_classify(
        &self,
        instruction: String,
        text: String,
    ) -> impl Future<Output = Result<bool, ClassifyError>> + Send;

    fn generate_text(
        &self,
        instruction: String,
        text: String,
    ) -> impl Future<Output = Result<String, GenerateTextError>> + Send;

    fn score_float(
        &self,
        instruction: String,
        text: String,
        min_bound: f64,
        max_bound: f64,
    ) -> impl Future<Output = Result<f64, ScoreFloatError>> + Send;

    fn score_int(
        &self,
        instruction: String,
        text: String,
        min_bound: i64,
        max_bound: i64,
    ) -> impl Future<Output = Result<i64, ScoreIntError>> + Send;

    fn parse<T>(&self, text: String) -> impl Future<Output = Result<T, ParseError>> + Send
    where
        T: for<'de> Deserialize<'de> + JsonSchema;
}

pub struct OpenAIModel {
    model: String,
    api_key: String,
}

#[derive(Debug, Clone)]
enum MessageRole {
    System,
    Assistant,
    User,
}

impl Serialize for MessageRole {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        match self {
            MessageRole::System => serializer.serialize_str("system"),
            MessageRole::Assistant => serializer.serialize_str("assistant"),
            MessageRole::User => serializer.serialize_str("user"),
        }
    }
}

impl<'de> Deserialize<'de> for MessageRole {
    fn deserialize<D>(deserializer: D) -> Result<MessageRole, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "system" => Ok(MessageRole::System),
            "assistant" => Ok(MessageRole::Assistant),
            "user" => Ok(MessageRole::User),
            _ => Err(serde::de::Error::custom("invalid role")),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Message {
    role: MessageRole,
    content: String,
    obj: Option<Map<String, Value>>,
}

#[derive(Debug, Serialize, Clone)]
struct OpenAIMessage {
    role: MessageRole,
    content: String,
}

impl<'de> Deserialize<'de> for OpenAIMessage {
    fn deserialize<D>(deserializer: D) -> Result<OpenAIMessage, D::Error>
    where
        D: serde::de::Deserializer<'de>,
    {
        let obj = Map::<String, Value>::deserialize(deserializer)?;
        let role = obj
            .get("role")
            .ok_or(serde::de::Error::custom("role not found"))?;
        let content = obj
            .get("content")
            .ok_or(serde::de::Error::custom("content not found"))?;
        Ok(OpenAIMessage {
            role: serde_json::from_value(role.clone()).unwrap(),
            content: serde_json::from_value(content.clone()).unwrap(),
        })
    }
}

struct GenerateMessageOptions {
    temperature: f64,
    force_json: bool,
}

struct GenerateMessageOptionsBuilder {
    temperature: f64,
    force_json: bool,
}

impl GenerateMessageOptionsBuilder {
    pub fn new() -> Self {
        GenerateMessageOptionsBuilder {
            temperature: 0.0,
            force_json: false,
        }
    }

    pub fn temperature(&mut self, temperature: f64) -> &mut Self {
        self.temperature = temperature;
        self
    }

    pub fn force_json(&mut self, force_json: bool) -> &mut Self {
        self.force_json = force_json;
        self
    }

    pub fn build(&self) -> GenerateMessageOptions {
        GenerateMessageOptions {
            temperature: self.temperature,
            force_json: self.force_json,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatRequestBody {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f64,
    response_format: ResponseFormat,
}

#[derive(Debug, Serialize, Deserialize)]
struct ResponseFormat {
    r#type: ResponseFormatType,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ResponseFormatType {
    JsonObject,
    Text,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Choice {
    message: OpenAIMessage,
}

impl OpenAIModel {
    pub fn new(model: String) -> Self {
        if let Ok(api_key) = std::env::var(OPENAI_API_KEY_NAME) {
            OpenAIModel { model, api_key }
        } else {
            panic!("{} not found in environment variables", OPENAI_API_KEY_NAME);
        }
    }

    async fn generate_message(
        &self,
        messages: Vec<Message>,
        options: GenerateMessageOptions,
    ) -> Result<Message, ChatError> {
        let url = format!("https://{}{}", OPENAI_API_BASE, OPENAI_API_CHAT_ENDPOINT);
        let client = reqwest::Client::new();
        let response_format_type = if options.force_json {
            ResponseFormatType::JsonObject
        } else {
            ResponseFormatType::Text
        };
        let openai_messages: Vec<OpenAIMessage> = messages
            .iter()
            .map(|message| OpenAIMessage {
                role: message.role.clone(),
                content: message.content.clone(),
            })
            .collect();
        let body = ChatRequestBody {
            model: self.model.clone(),
            messages: openai_messages,
            temperature: options.temperature,
            response_format: ResponseFormat {
                r#type: response_format_type,
            },
        };
        let response = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await;
        match response {
            Ok(response) => {
                if response.status() != reqwest::StatusCode::OK {
                    return Err(ChatError {
                        message: format!(
                            "{}: {}",
                            response.status(),
                            response.text().await.unwrap()
                        ),
                    });
                }
                match response.json::<ChatResponse>().await {
                    Ok(chat_response) => {
                        if let Some(choice) = chat_response.choices.first() {
                            let message = choice.message.clone();
                            if options.force_json {
                                let obj =
                                    serde_json::from_str::<Map<String, Value>>(&message.content);
                                if let Ok(obj) = obj {
                                    return Ok(Message {
                                        role: message.role,
                                        content: message.content,
                                        obj: Some(obj),
                                    });
                                } else {
                                    return Err(ChatError {
                                        message: String::from("Failed to parse response"),
                                    });
                                }
                            } else {
                                return Ok(Message {
                                    role: message.role,
                                    content: message.content,
                                    obj: None,
                                });
                            }
                        } else {
                            return Err(ChatError {
                                message: String::from("Choice not found in response"),
                            });
                        }
                    }
                    Err(e) => {
                        return Err(ChatError {
                            message: String::from("Failed to parse response, error: ")
                                + &e.to_string(),
                        });
                    }
                }
            }
            Err(e) => {
                return Err(ChatError {
                    message: format!("{}", e),
                });
            }
        };
    }
}

impl Model for OpenAIModel {
    async fn classify(
        &self,
        instruction: String,
        text: String,
        choices: Vec<String>,
    ) -> Result<usize, ClassifyError> {
        let (choices_display, lookup_table) = display_choices(choices);
        let input_text = format!(
            "Instruction:\n{}\n\nText:\n{}\n\nChoices:\n{}\n\nValid JSON:",
            instruction, text, choices_display
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: String::from("Classify the following text with the provided instruction and choices. To classify, provide the key of the choice:\n{\"classification\": string}\n\nFor example, if the correct choice is 'Z. description of choice Z', then provide 'Z' as the classification as valid JSON:\n{\"classification\": \"Z\"}"),
                obj: None,
            },
            Message {
                role: MessageRole::User,
                content: input_text,
                obj: None,
            },
        ];
        let options = GenerateMessageOptionsBuilder::new()
            .temperature(0.0)
            .force_json(true)
            .build();
        if let Ok(message) = self.generate_message(messages, options).await {
            if let Some(obj) = message.obj {
                if let Some(classification) = obj["classification"].as_str() {
                    if let Some(choice_index) = lookup_table.get(classification) {
                        return Ok(*choice_index);
                    } else {
                        return Err(ClassifyError {
                            message: format!("Invalid classification: {}", classification),
                        });
                    }
                } else {
                    return Err(ClassifyError {
                        message: String::from("Classification not found in response"),
                    });
                }
            } else {
                return Err(ClassifyError {
                    message: String::from("Object not found in response"),
                });
            }
        } else {
            return Err(ClassifyError {
                message: String::from("Failed to generate message"),
            });
        }
    }

    async fn binary_classify(
        &self,
        instruction: String,
        text: String,
    ) -> Result<bool, ClassifyError> {
        self.classify(
            instruction,
            text,
            vec!["true".to_string(), "false".to_string()],
        )
        .await
        .map(|index| index == 0)
    }

    async fn generate_text(
        &self,
        instruction: String,
        text: String,
    ) -> Result<String, GenerateTextError> {
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: instruction,
                obj: None,
            },
            Message {
                role: MessageRole::User,
                content: text,
                obj: None,
            },
        ];
        let options = GenerateMessageOptionsBuilder::new()
            .temperature(0.0)
            .force_json(false)
            .build();
        if let Ok(message) = self.generate_message(messages, options).await {
            Ok(message.content)
        } else {
            Err(GenerateTextError {
                message: String::from("Failed to generate message"),
            })
        }
    }

    async fn score_float(
        &self,
        instruction: String,
        text: String,
        min_bound: f64,
        max_bound: f64,
    ) -> Result<f64, ScoreFloatError> {
        let input_text = format!(
            "Instruction:\n{}\n\nText:\n{}\n\nRange:\n[{}, {}]\n\nValid JSON:",
            instruction, text, min_bound, max_bound
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: String::from("Score the following text with the provided instruction and range as a float value as valid JSON:\n{\"score\": float}"),
                obj: None,
            },
            Message {
                role: MessageRole::User,
                content: input_text,
                obj: None,
            },
        ];
        let options = GenerateMessageOptionsBuilder::new()
            .temperature(0.0)
            .force_json(true)
            .build();
        if let Ok(message) = self.generate_message(messages, options).await {
            if let Some(obj) = message.obj {
                if let Some(score) = obj["score"].as_f64() {
                    return Ok(score);
                } else {
                    return Err(ScoreFloatError {
                        message: String::from("Score not found in response"),
                    });
                }
            } else {
                return Err(ScoreFloatError {
                    message: String::from("Object not found in response"),
                });
            }
        } else {
            Err(ScoreFloatError {
                message: String::from("Failed to generate message"),
            })
        }
    }

    async fn score_int(
        &self,
        instruction: String,
        text: String,
        min_bound: i64,
        max_bound: i64,
    ) -> Result<i64, ScoreIntError> {
        let input_text = format!(
            "Instruction:\n{}\n\nText:\n{}\n\nRange:\n[{}, {}]\n\nValid JSON:",
            instruction, text, min_bound, max_bound
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: String::from("Score the following text with the provided instruction and range as an integer value as valid JSON:\n{\"score\": int}"),
                obj: None,
            },
            Message {
                role: MessageRole::User,
                content: input_text,
                obj: None,
            },
        ];
        let options = GenerateMessageOptionsBuilder::new()
            .temperature(0.0)
            .force_json(true)
            .build();
        if let Ok(message) = self.generate_message(messages, options).await {
            if let Some(obj) = message.obj {
                if let Some(score) = obj["score"].as_i64() {
                    return Ok(score);
                } else {
                    return Err(ScoreIntError {
                        message: String::from("Score not found in response"),
                    });
                }
            } else {
                return Err(ScoreIntError {
                    message: String::from("Object not found in response"),
                });
            }
        } else {
            Err(ScoreIntError {
                message: String::from("Failed to generate message"),
            })
        }
    }

    async fn parse<T>(&self, text: String) -> Result<T, ParseError>
    where
        T: for<'de> Deserialize<'de> + JsonSchema,
    {
        let json_schema_string = struct_to_json_schema_string::<T>();
        let input_text = format!(
            "Text:\n{}\n\nSchema:\n{}\n\nValid JSON:",
            text, json_schema_string
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: String::from("Parse the following text with the provided schema."),
                obj: None,
            },
            Message {
                role: MessageRole::User,
                content: input_text,
                obj: None,
            },
        ];
        let options = GenerateMessageOptionsBuilder::new()
            .temperature(0.0)
            .force_json(true)
            .build();
        match self.generate_message(messages, options).await {
            Ok(message) => {
                if let Some(obj) = message.obj {
                    if let Ok(parsed_obj) = json_response_to_obj::<T>(obj) {
                        return Ok(parsed_obj);
                    } else {
                        return Err(ParseError {
                            message: String::from("Failed to parse response"),
                        });
                    }
                } else {
                    return Err(ParseError {
                        message: String::from("Object not found in response"),
                    });
                }
            }
            Err(e) => {
                return Err(ParseError {
                    message: format!("{}", e),
                });
            }
        }
    }
}

fn struct_to_json_schema_string<T: JsonSchema>() -> String {
    let schema = schema_for!(T);
    to_string_pretty(&schema).unwrap()
}

fn json_response_to_obj<T>(json_response: Map<String, Value>) -> Result<T, Error>
where
    T: for<'de> Deserialize<'de> + JsonSchema,
{
    let json_str = to_string(&json_response).unwrap();
    let obj = serde_json::from_str::<T>(&json_str);
    if let Ok(obj) = obj {
        Ok(obj)
    } else {
        Err(Error)
    }
}

fn display_choices(choices: Vec<String>) -> (String, HashMap<String, usize>) {
    let mut choices_displays = vec![];
    let mut decode_map: HashMap<String, usize> = HashMap::new();
    for (i, choice) in choices.iter().enumerate() {
        let label = index_to_alpha(i);
        choices_displays.push(format!("{}. {}", label, choice));
        decode_map.insert(label, i);
    }
    (choices_displays.join("\n"), decode_map)
}

fn index_to_alpha(mut index: usize) -> String {
    let mut alpha = String::new();
    loop {
        alpha = format!("{}{}", ('A' as u8 + (index % 26) as u8) as char, alpha);
        index = index / 26 - 1;
        if index == 0 {
            break;
        }
    }
    alpha
}

#[derive(Debug, Clone)]
pub struct ClassifyError {
    message: String,
}

impl std::fmt::Display for ClassifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ClassifyError: {}", self.message)
    }
}

#[derive(Debug, Clone)]
pub struct GenerateTextError {
    message: String,
}

impl std::fmt::Display for GenerateTextError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "GenerateTextError: {}", self.message)
    }
}

#[derive(Debug, Clone)]
pub struct ScoreFloatError {
    message: String,
}

impl std::fmt::Display for ScoreFloatError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ScoreFloatError: {}", self.message)
    }
}

#[derive(Debug, Clone)]
pub struct ScoreIntError {
    message: String,
}

impl std::fmt::Display for ScoreIntError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ScoreIntError: {}", self.message)
    }
}

#[derive(Debug, Clone)]
pub struct ParseError {
    message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ParseError: {}", self.message)
    }
}

pub struct ChatError {
    message: String,
}

impl std::fmt::Display for ChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ChatError: {}", self.message)
    }
}
