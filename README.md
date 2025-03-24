# ID Classifier Application

This application provides ID document classification using machine learning. It consists of a frontend, backend API, and model service components.

## Prerequisites

- Docker and Docker Compose
- OpenAI API key

## Quick Start

### 1. Set up your OpenAI API key

Create a `.env` file in the `model/inference` directory with your OpenAI API key:

OPENAI_API_KEY=your_openai_api_key_here

Never commit your API key to version control. The `.env` files are included in `.gitignore`.

### 2. Build and run with Docker

From the project root directory, run:

```bash
docker-compose up -d
```

This will:
- Build all necessary containers
- Start the frontend, backend, and model services
- Configure networking between services

### 3. Access the application

- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- Model Service: http://localhost:8000

## Development

### Service Architecture

- **Frontend (React)**: Provides the user interface for uploading and viewing ID documents
- **Backend (FastAPI)**: Handles file uploads and communicates with the model service
- **Model Service (FastAPI)**: Processes images and performs ID classification

### Environment Variables

Additional environment variables can be configured in the `docker-compose.yaml` file for each service.

### Project Structure

```
├── frontend/               # React frontend application
│   ├── public/             # Static files
│   ├── src/                # Source code
│   ├── Dockerfile          # Frontend container configuration
│   └── package.json        # Dependencies
│
├── backend/                # FastAPI backend service
│   ├── app/                # Application code
│   │   ├── api/            # API routes
│   │   ├── core/           # Core functionality
│   │   └── models/         # Data models
│   ├── Dockerfile          # Backend container configuration
│   └── requirements.txt    # Python dependencies
│
├── model/                  # Model service
│   ├── inference/          # Inference code
│   │   ├── app.py          # FastAPI application
│   │   └── .env            # Environment variables (not in version control)
│   ├── Dockerfile          # Model service container configuration
│   └── requirements.txt    # Python dependencies
│
└── docker-compose.yaml     # Multi-container configuration
```

### Communication Flow

```
┌─────────────┐     HTTP      ┌─────────────┐     HTTP      ┌─────────────┐     API      ┌─────────────┐ API  ┌─────────────┐
│             │    Request    │             │    Request    │             │    Request   │             │─────►│     CNN     │
│    User     │ ───────────►  │   Frontend  │ ───────────►  │   Backend   │ ───────────► │    Model    │◄─────│   (FastAPI) │
│  (Browser)  │               │   (React)   │               │  (FastAPI)  │              │   Service   │      └─────────────┘
│             │               │             │               │             │              │  (FastAPI)  │      ┌─────────────┐
│             │               │             │               │             │              │             │─────►│     LLM     │
│             │ ◄───────────  │             │ ◄───────────  │             │ ◄─────────── │             │◄─────│  (FastAPI)  │
└─────────────┘     HTTP      └─────────────┘     HTTP      └─────────────┘     API      └─────────────┘  API └─────────────┘
                   Response                      Response                      Response              
                                                                                                        
```

1. User uploads an ID document through the frontend interface
2. Frontend sends the image to the backend API
3. Backend sends the image to the model service for classification
4. Model service processes the image using CNN for feature extraction
5. Extracted features are sent to the LLM for final classification
6. Results are passed back through the chain to the user interface

## Troubleshooting

1. Verify your OpenAI API key
2. Check the logs for each service
3. Ensure all ports are available

