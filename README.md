# mlops-pytorch-pipeline
mlops assigment 3 da25m582 Kunal Sakhuja
# End-to-End MLOps: PyTorch Training & Serving on Kubernetes

A complete, containerized Machine Learning pipeline that trains a ResNet18 model on the CIFAR-10 dataset and deploys it as a REST API. The entire infrastructure runs on Kubernetes, utilizing Persistent Volumes to seamlessly share model weights between the training job and the serving deployment.

## Architecture Overview

1. **Training (Kubernetes Job):** A PyTorch script trains ResNet18 on CIFAR-10. It uses a Kubernetes `ConfigMap` for hyperparameters and saves the final model checkpoint (`classifier_v1.pt`) to a Persistent Volume Claim (PVC).
2. **Storage (PVC):** A shared virtual hard drive that decouples model training from model serving.
3. **Serving (Kubernetes Deployment & Service):** A FastAPI application mounts the specific `checkpoints` sub-path of the PVC, loads the model into memory, and exposes a `/predict` endpoint to classify uploaded images.

## Tech Stack
* **Machine Learning:** PyTorch, Torchvision
* **API Framework:** FastAPI, Uvicorn
* **Containerization:** Docker
* **Orchestration:** Kubernetes (Kind for local development)

## Project Structure

```text
.
├── configs/
│   └── training_config.yaml      # Hyperparameters and paths
├── docker/
│   ├── Dockerfile.train          # Image build for training job
│   └── Dockerfile.serve          # Image build for FastAPI server
├── k8s/
│   ├── configmap.yaml            # Injects training_config.yaml into pods
│   ├── pvc.yaml                  # Persistent storage allocation
│   ├── training-job.yaml         # One-off K8s Job for model training
│   └── serving-deployment.yaml   # FastAPI Deployment and LoadBalancer/Service
├── requirements/
│   ├── train.txt                 # Training dependencies
│   └── serve.txt                 # Serving dependencies
└── src/
    ├── dataset.py                # CIFAR-10 data loaders
    ├── model.py                  # ResNet18 architecture setup
    ├── train.py                  # Training loop
    └── serve.py                  # FastAPI endpoints
Prerequisites
To run this project locally, you need:

Docker

Kind (Kubernetes IN Docker)

kubectl

Quick Start Guide
1. Build and Load Docker Images
Since we are using kind, we must build the images locally and load them directly into the cluster's registry.
# Build the images
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Load into Kind cluster (assuming your cluster is named 'desktop' or 'kind')
kind load docker-image mlops-train:v1 --name <your-cluster-name>
kind load docker-image mlops-serve:v1 --name <your-cluster-name>

# Alternatively, import directly via containerd:
# docker save mlops-train:v1 | docker exec -i <control-plane-container> ctr -n k8s.io images import -
2. Setup Kubernetes Infrastructure
Create the persistent volume and inject the configuration map.
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
3. Run the Training Job
Launch the PyTorch training job. The container will download CIFAR-10, train the model, and output the .pt file to the PVC.
kubectl apply -f k8s/training-job.yaml

# Monitor the training logs:
kubectl logs -f job/model-training -n ml-training

4. Deploy the Serving API
Once training is complete and the model is saved to the PVC, spin up the FastAPI server.
kubectl apply -f k8s/serving-deployment.yaml

# Check when the pods are 'Running'
kubectl get pods -n ml-training -w
5. Test the API
Forward the Kubernetes service port to your local machine:
kubectl port-forward svc/model-serving 8080:80 -n ml-training
Open a new terminal window, download a test image, and send it to the /predict endpoint:
# Download a random 200x200 image
curl -L -o test_image.jpg [https://picsum.photos/200](https://picsum.photos/200)

# Send for classification
curl -X POST http://localhost:8080/predict -F "file=@test_image.jpg"
Expected Output:

JSON
{
  "predicted_class": "dog",
  "confidence": 0.8932,
  "probabilities": {
    "airplane": 0.01,
    "automobile": 0.005,
    "bird": 0.02,
    "cat": 0.05,
    "deer": 0.01,
    "dog": 0.8932,
    "frog": 0.001,
    "horse": 0.01,
    "ship": 0.0005,
    "truck": 0.0003
  }
}