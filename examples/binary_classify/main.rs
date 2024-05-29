use llm_primitives::Model;
use llm_primitives::OpenAIModel;
use tokio;

#[tokio::main]
async fn main() {
    let model = OpenAIModel::new(String::from("gpt-4o"));
    let response = model
        .binary_classify(
            String::from("Determine if the text is positive"),
            String::from("I hate this product"),
        )
        .await;
    if let Ok(text) = response {
        println!("{}", text);
    } else {
        println!("{:?}", response);
    }
}
