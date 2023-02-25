const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const modelUrl =
  'https://tfhub.dev/tensorflow/tfjs-model/ssdlite_mobilenet_v2/1/default/1';

let model;

//load COCO-SSD graph model from TensorFlow Hub
//this model expects a 4-D tensor with image pixel values
const loadModel = async function () {
  console.log(`loading model from ${modelUrl}`);

  model = await tf.loadGraphModel(modelUrl, {fromTFHub: true});

  return model;
}

// preprocessing image before passing it to the graph model
// convert image to the appropriate Tensor
const processInput = function (imagePath) {
  console.log(`preprocessing image ${imagePath}`);

  const image = fs.readFileSync(imagePath);
  const buf = Buffer.from(image);
  const uint8array = new Uint8Array(buf);

  // convert an image into a 3-d tensor using the decodeImage() API
  // increase dimension to 4-d using expandDims()
  return tf.node.decodeImage(uint8array, 3).expandDims();
}

const runModel = function (inputTensor) {
  console.log('running model');

  return model.executeAsync(inputTensor);
}

// run
if (process.argv.length < 3) {
  console.log('please pass an image to process. ex:');
  console.log('  node run-tfjs-model.js /path/to/image.jpg');
} else {
  // e.g., /path/to/image.jpg
  let imagePath = process.argv[2];

  loadModel().then(model => {
    const inputTensor = processInput(imagePath);
    return runModel(inputTensor);
  }).then(prediction => {
    console.log(prediction);
  })
}
