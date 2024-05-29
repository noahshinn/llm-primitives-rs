use llm_primitives::Model;
use llm_primitives::OpenAIModel;
use tokio;

#[tokio::main]
async fn main() {
    let model = OpenAIModel::new(String::from("gpt-4o"));
    let response = model
        .generate_text(
            String::from("Respond to the user"),
            String::from("User: Hello, how are you?"),
        )
        .await;
    if let Ok(text) = response {
        println!("{}", text);
    } else {
        println!("{:?}", response);
    }
}
