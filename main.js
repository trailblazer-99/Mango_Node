import express from "express";
import cors from "cors";
import multer from "multer";
import sharp from "sharp";
import tf from "@tensorflow/tfjs-node";
import fs from "fs/promises";

const app = express();
const port = 8010;

app.use(cors());
app.use(express.json());

// Load TensorFlow Model
const model = await tf.loadLayersModel("file://model_mango.h5"); 

// File Handling using Multer
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Class Names
const CLASS_NAMES = [
  "Anthracnose",
  "Bacterial Canker",
  "Cutting Weevil",
  "Die Back",
  "Gall Midge",
  "Healthy",
  "Powdery Mildew",
  "Sooty Mould",
];

// Ping Route (Health Check)
app.get("/ping", (req, res) => {
  res.send("Mango Disease Detection");
});

// Predict Route
app.post("/predict", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).send("No file uploaded.");
    }

    const imageBuffer = req.file.buffer;

    // Preprocess Image using Sharp
    const imageTensor = tf.node.decodeImage(imageBuffer); 
    const resized = tf.image.resizeBilinear(imageTensor, [180, 180]);
    const normalized = resized.div(255); // Normalize
    const batched = normalized.expandDims(0); // Create a batch

    // Make Prediction
    const prediction = model.predict(batched);
    const predictedIndex = prediction.argMax(-1).dataSync()[0];
    const confidence = prediction.max().dataSync()[0];

    res.json({
      class: CLASS_NAMES[predictedIndex],
      confidence: (confidence * 100).toFixed(2), // As percentage
    });

  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});