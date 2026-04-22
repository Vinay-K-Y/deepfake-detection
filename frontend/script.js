const API_URL = "http://127.0.0.1:8000/predict";

const input = document.getElementById("imageInput");
const preview = document.getElementById("previewImage");

input.addEventListener("change", () => {
    const file = input.files[0];
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
    }
});

async function uploadImage() {
    const file = input.files[0];
    if (!file) {
        alert("Please select an image");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("result").innerText = "⏳ Processing...";

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        document.getElementById("result").innerHTML = `
            Prediction: <span style="color:${data.prediction === "FAKE" ? "red" : "lightgreen"}">
                ${data.prediction}
            </span><br>
            Confidence: ${(data.confidence * 100).toFixed(2)}%
        `;
    } catch (error) {
        document.getElementById("result").innerText = "❌ Error connecting to API";
        console.error(error);
    }
}