imageUpload.addEventListener("change", async (event) => {

    event.preventDefault();        // ⛔ stop page reload
    event.stopPropagation();       // ⛔ block element bubbling

    const file = imageUpload.files[0];
    if (!file) return;

    resultCard.classList.remove("hidden");

    previewImage.src = URL.createObjectURL(file);
    foodName.innerText = "⚡ SCANNING...";

    const fd = new FormData();
    fd.append("file", file);

    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: fd
        });

        const data = await res.json();

        foodName.innerText = data.prediction.toUpperCase();
        foodType.innerText = data.category;
        calories.innerText = data.calories;
        benefitText.innerText = data.advice;

    } catch (error) {
        foodName.innerText = "FAILED!";
    }
});
