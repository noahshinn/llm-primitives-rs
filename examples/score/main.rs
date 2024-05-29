use llm_primitives::Model;
use llm_primitives::OpenAIModel;
use tokio;

#[tokio::main]
async fn main() {
    let model = OpenAIModel::new(String::from("gpt-4o"));
    let response = model
        .score_int(
            String::from("Score the user's review from (1) bad to (5) good"),
            String::from("The product was great!"),
            1,
            5,
        )
        .await;
    if let Ok(text) = response {
        println!("{}", text);
    } else {
        println!("{:?}", response);
    }
}
