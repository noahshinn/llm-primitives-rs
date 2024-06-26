use llm_primitives::Model;
use llm_primitives::OpenAIModel;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio;

#[derive(JsonSchema, Deserialize, Debug)]
struct Address {
    street: String,
    number: i64,
}

#[tokio::main]
async fn main() {
    let model = OpenAIModel::new(String::from("gpt-4o"));
    let response = model
        .parse::<Address>(String::from("My street is 123 main st"))
        .await;
    if let Ok(address) = response {
        println!(
            "Street name: {}\nStreet number: {}",
            address.street, address.number
        );
    } else {
        println!("{:?}", response);
    }
}
