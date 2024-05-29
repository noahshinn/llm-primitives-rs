use std::vec;

use llm_primitives::Model;
use llm_primitives::OpenAIModel;
use tokio;

#[tokio::main]
async fn main() {
    let model = OpenAIModel::new(String::from("gpt-4o"));
    let response = model
        .classify(
            String::from("Determine the sentiment of the text"),
            String::from("I love this product"),
            vec![
                "Positive".to_string(),
                "Negative".to_string(),
                "Neutral".to_string(),
            ],
        )
        .await;
    if let Ok(text) = response {
        println!("{}", text);
    } else {
        println!("{:?}", response);
    }
}
