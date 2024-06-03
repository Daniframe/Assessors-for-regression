from transformers import AutoTokenizer, AutoModel
import torch
import json

with open("afrla_models.json") as f:
    data = json.load(f)

model_strings = [json.dumps(model, indent = 2) for model in data["descriptions"]]

embedder = "bert-base-uncased"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(embedder)
model = AutoModel.from_pretrained(embedder)
model = model.to(device)

def generate_embeddings(model, model_strings):
    inputs = tokenizer(model_strings, return_tensors = "pt", padding = True, truncation = True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the token embeddings as the embedding for the entire input
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

embeddings = generate_embeddings(model, model_strings).cpu().numpy()
print(embeddings.shape)

# Save the embeddings
model_embeddings = {
    "names" : data["labels"],
    "embeddings" : embeddings.tolist()
}

with open("afrla_embeddings.json", "w") as f:
    json.dump(model_embeddings, f, indent = 4)