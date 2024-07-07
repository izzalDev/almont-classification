async function getImageTensorFromPath(path, dims = [1, 3, 128, 128]) {
    // 1. load the image
    var image = await loadImagefromPath(path, dims[2], dims[3]);
    // 2. convert to tensor
    var imageTensor = imageDataToTensor(image, dims);
    // 3. return the tensor
    return imageTensor;
}

async function loadImagefromPath(path, width = 128, height = 128) {
    // Use Jimp to load the image and resize it.
    var imageData = await Jimp.read(path).then((imageBuffer) => {
        return imageBuffer.resize(width, height);
    });

    return imageData;
}

function imageDataToTensor(image, dims) {
    const [n, c, h, w] = dims;
    const imageData = new Float32Array(n * c * h * w);
    const data = image.bitmap.data;

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
        for (let d = 0; d < c; d++) {
            imageData[d * h * w + y * w + x] = data[(y * w + x) * 4 + d] / 255.0;
        }
        }
    }

    return new ort.Tensor('float32', imageData, dims);
}