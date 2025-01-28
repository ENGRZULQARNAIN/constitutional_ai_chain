# ConstitutionalChain

ConstitutionalChain is a framework designed to implement and experiment with constitutional AI principles in building robust, ethical, and interpretable AI systems. This repository leverages cutting-edge Generative AI techniques and chains them together with predefined constitutions, ensuring alignment with human values and desired outcomes.

## Key Features

- **Constitutional Principles**: Integrates AI models with rules and principles to ensure ethical outputs.
- **Generative AI Integration**: Combines generative capabilities with constitutional logic for enhanced decision-making.
- **Tool Support**: Supports various tools for RAG (Retrieval-Augmented Generation), Agents, and Prompt Engineering.
- **Extensibility**: Easily extendable for custom constitutions and new use cases.
- **FastAPI Deployment**: Out-of-the-box support for deploying on FastAPI for production use cases.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/engrzulqarnain/constitutional_ai_chain.git
   cd ConstitutionalChain
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate # For Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start
1. Define your constitution in the `constitutions/` directory.
2. Configure the AI chain in `config/chain_config.yaml`.
3. Run the main script:
   ```bash
   python main.py
   ```
4. (Optional) Deploy the API:
   ```bash
   uvicorn app.main:app --reload
   ```

### Example
```python
from constitutional_chain import ConstitutionalChain

# Define a simple constitution
constitution = [
    "The AI must avoid generating harmful content.",
    "The AI should prioritize clarity and accuracy in its responses.",
]

# Initialize the chain
chain = ConstitutionalChain(constitution=constitution)

# Input for testing
response = chain.run("Explain the concept of Constitutional AI.")
print(response)
```

## Directory Structure
```
ConstitutionalChain/
├── app/                 # FastAPI application
├── config/              # Configuration files
├── constitutions/       # Predefined constitutions
├── examples/            # Usage examples
├── constitutional_chain/ # Core library
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── main.py              # Entry point
└── README.md            # Documentation
```

## Tags
- Constitutional AI
- Generative AI
- Ethical AI
- NLP
- AI Alignment
- FastAPI
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)
- AI Agents

## Contributing
We welcome contributions! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License
This project is licensed under the [MIT License](LICENSE).
