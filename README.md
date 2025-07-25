# DATAGEN (Previously AI-Data-Analysis-MultiAgent)

![DATAGEN Banner](DATAGEN.jpg "DATAGEN Banner")

## About DATAGEN
DATAGEN is a powerful brand name that represents our vision of leveraging artificial intelligence technology for data generation and analysis. The name combines "DATA" and "GEN"(generation), perfectly embodying the core functionality of this project - automated data analysis and research through a multi-agent system.

Visit us at [DATAGEN Digital](https://datagen.digital/)(website under development) to learn more about our vision and services.

![System Architecture](Architecture.png)
## Overview

DATAGEN is an advanced AI-powered data analysis and research platform that utilizes multiple specialized agents to streamline tasks such as data analysis, visualization, and report generation. Our platform leverages cutting-edge technologies including LangChain, OpenAI's GPT models, and LangGraph to handle complex research processes, integrating diverse AI architectures for optimal performance.

## Key Features

### Intelligent Analysis Core
- **Advanced Hypothesis Engine**
  - AI-driven hypothesis generation and validation
  - Automated research direction optimization
  - Real-time hypothesis refinement
- **Enterprise Data Processing**
  - Robust data cleaning and transformation
  - Scalable analysis pipelines
  - Automated quality assurance
- **Dynamic Visualization Suite**
  - Interactive data visualization
  - Custom report generation
  - Automated insight extraction

### Advanced Technical Architecture
- **Multi-Agent Intelligence** 
  - Specialized agents for diverse tasks
  - Intelligent task distribution
  - Real-time coordination and optimization
- **Smart Memory Management**
  - State-of-the-art Note Taker agent
  - Efficient context retention system
  - Seamless workflow integration
- **Adaptive Processing Pipeline**
  - Dynamic workflow adjustment
  - Automated resource optimization
  - Real-time performance monitoring

## Why DATAGEN Stands Out

DATAGEN revolutionizes data analysis through its innovative multi-agent architecture and intelligent automation capabilities:

1. **Advanced Multi-Agent System**
   - Specialized agents working in harmony
   - Intelligent task distribution and coordination
   - Real-time adaptation to complex analysis requirements

2. **Smart Context Management**
   - Pioneering Note Taker agent for state tracking
   - Efficient memory utilization and context retention
   - Seamless integration across analysis phases

3. **Enterprise-Grade Performance**
   - Robust and scalable architecture
   - Consistent and reliable outcomes
   - Production-ready implementation

## System Requirements

- Python 3.10 or higher
- Jupyter Notebook environment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/starpig1129/DATAGEN.git
```
2. Create and activate a Conda virtual environment:
```bash
conda create -n data_assistant python=3.10
conda activate data_assistant
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Set up environment variables:
**Rename `.env Example` to `.env` and fill all the values**
```sh
# Your data storage path(required)
DATA_STORAGE_PATH =./data_storage/

# Anaconda installation path(required)
CONDA_PATH = /home/user/anaconda3

# Conda environment name(required)
CONDA_ENV = envname

# ChromeDriver executable path(required)
CHROMEDRIVER_PATH =./chromedriver-linux64/chromedriver

# Firecrawl API key (optional)
# Note: If this key is missing, query capabilities may be reduced
FIRECRAWL_API_KEY = XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# OpenAI API key (required)
# Warning: This key is essential; the program will not run without it
OPENAI_API_KEY = XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# LangChain API key (optional)
# Used for monitoring the processing
LANGCHAIN_API_KEY = XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```
## Usage

1. Start Jupyter Notebook *(optional)*.
2. Place your data file (e.g., `data.csv`) in `data_storage`.
3. **Notebook workflow**: open `main.ipynb`, run all cells, and modify `userInput` in the last cell to start the analysis.
4. **CLI workflow**: run `python workflow.py "your research instructions"` to execute the entire process from the command line.
=======
### Using Jupyter Notebook

1. Start Jupyter Notebook:

2. Set YourDataName.csv in data_storage

3. Open the `main.ipynb` file.

4. Run all cells to initialize the system and create the workflow.

5. In the last cell, you can customize the research task by modifying the `userInput` variable.

6. Run the final few cells to execute the research process and view the results.

### Using Python Script

You can also run the system directly using main.py:

1. Place your data file (e.g., YourDataName.csv) in the data_storage directory

2. Run the script:
```bash
python main.py
```

3. By default, it will process 'OnlineSalesData.csv'. To analyze a different dataset, modify the user_input variable in the main() function of main.py:
```python
user_input = '''
datapath:YourDataName.csv
Use machine learning to perform data analysis and write complete graphical reports
'''
```

## Main Components

- `hypothesis_agent`: Generates research hypotheses
- `process_agent`: Supervises the entire research process
- `visualization_agent`: Creates data visualizations
- `code_agent`: Writes data analysis code
- `searcher_agent`: Conducts literature and web searches
- `report_agent`: Writes research reports
- `quality_review_agent`: Performs quality reviews
- `note_agent`: Records the research process

## Workflow

The system uses LangGraph to create a state graph that manages the entire research process. The workflow includes the following steps:

1. Hypothesis generation
2. Human choice (continue or regenerate hypothesis)
3. Processing (including data analysis, visualization, search, and report writing)
4. Quality review
5. Revision as needed

## Customization

You can customize the system behavior by modifying the agent creation and workflow definition in `main.ipynb`.
All agent prompts are stored in the `prompts/` directory for easier editing.

## Notes

- Ensure you have sufficient OpenAI API credits, as the system will make multiple API calls.
- The system may take some time to complete the entire research process, depending on the complexity of the task.
- **WARNING**: The agent system may modify the data being analyzed. It is highly recommended to backup your data before using this system.
## Current Issues and Solutions
1. OpenAI Internal Server Error (Error code: 500)
2. NoteTaker Efficiency Improvement
3. Overall Runtime Optimization
4. Refiner needs to be better
## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Strategic Partnership
[![CTL GROUP](https://img.shields.io/badge/DATAGEN-Strategic_Partner-blue)](https://datagen.digital/)

We are excited to announce our upcoming strategic partnership with CTL GROUP, an innovative AI-Powered Crypto Intelligence Platform currently in development. This collaboration will bring together advanced AI research capabilities with crypto market intelligence:

### Upcoming Partnership Features
- **AI Crypto Research Integration**
  - Automated market research and analysis system
  - Advanced whale tracking capabilities
  - Real-time sentiment analysis tools
  - Comprehensive trading insights and strategies

- **Platform Features** (Coming Soon)
  - State-of-the-art AI-powered crypto insights
  - Smart trading strategy development
  - Advanced whale & on-chain activity monitoring
  - Interactive community engagement tools

- **Token Integration Benefits** (Coming Soon)
  - Dynamic staking rewards system
  - Premium tools and features access
  - Innovative passive income opportunities
  - Exclusive platform privileges

The platform is currently under development. Follow our progress on [GitHub](https://github.com/ctlgroupdev).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=starpig1129/DATAGEN&type=Date)](https://star-history.com/#starpig1129/DATAGEN&Date)

## Other Projects
Here are some of my other notable projects:
### ShareLMAPI
ShareLMAPI is a local language model sharing API that uses FastAPI to provide interfaces, allowing different programs or device to share the same local model, thereby reducing resource consumption. It supports streaming generation and various model configuration methods.
- GitHub: [ShareLMAPI](https://github.com/starpig1129/ShareLMAPI)
### PigPig: Advanced Multi-modal LLM Discord Bot: 
A powerful Discord bot based on multi-modal Large Language Models (LLM), designed to interact with users through natural language. 
It combines advanced AI capabilities with practical features, offering a rich experience for Discord communities.
- GitHub: [ai-discord-bot-PigPig](https://github.com/starpig1129/ai-discord-bot-PigPig)
