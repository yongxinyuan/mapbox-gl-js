export function drawInCanvas(image, width, height) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.width = width;
    canvas.height = height;

    const pixelNum = image.length / 4;

    for (let i = 0, len = image.length; i < len; i = i + 4) {
        const r = image[i];
        const g = image[i + 1];
        const b = image[i + 2];
        const a = image[i + 3];

        const pixelIndex = i / 4;

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a})`;

        ctx.fillRect(
            pixelIndex % width,
            Math.floor(pixelIndex / width),
            1,
            1
        );
    }

    document.body.style.overflow = "auto";
    document.body.appendChild(canvas);


}