from fastapi import FastAPI

model.to(DEVICE)
model.eval()

# =============================================================================
# Request Schema
# =============================================================================

class PromptRequest(BaseModel):
    prompt: str

# =============================================================================
# Prediction Function
# =============================================================================

def predict_prompt(prompt_text):

    encoded_input = tokenizer(
        CLASSIFICATION_PREFIX + prompt_text,
        max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            input_ids=encoded_input["input_ids"].to(DEVICE),
            attention_mask=encoded_input["attention_mask"].to(DEVICE)
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    harmful_probability = probabilities[1].item()
    benign_probability = probabilities[0].item()

    predicted_label = (
        "HARMFUL"
        if harmful_probability > 0.5
        else "BENIGN"
    )

    return {
        "label": predicted_label,
        "harmful_probability": harmful_probability,
        "benign_probability": benign_probability
    }

# =============================================================================
# API Endpoint
# =============================================================================

@app.post("/predict")
def predict(request: PromptRequest):

    result = predict_prompt(request.prompt)

    return {
        "prompt": request.prompt,
        "prediction": result
    }