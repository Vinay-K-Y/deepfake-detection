const API_URL = "http://127.0.0.1:8000/verify";

const input = document.getElementById("imageInput");
const preview = document.getElementById("previewImage");
const resultDiv = document.getElementById("result");

// Show preview
input.addEventListener("change", () => {
    const file = input.files[0];
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
    }
});

// Upload + Predict
async function uploadImage() {
    const file = input.files[0];

    if (!file) {
        alert("Please select an image");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    resultDiv.innerText = "⏳ Processing...";

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        // Handle HTTP errors
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        // Handle backend errors (like no face detected)
        if (data.error) {
            resultDiv.innerText = `⚠️ ${data.error}`;
            return;
        }

        // Display result (FIXED KEYS)
        resultDiv.innerHTML = `
            Prediction: <span style="color:${data.verdict === "FAKE" ? "red" : "lightgreen"}">
                ${data.verdict}
            </span><br>
            Confidence: ${data.confidence}
        `;

    } catch (error) {
        resultDiv.innerText = "❌ Error processing request";
        console.error("Error:", error);
    }
}