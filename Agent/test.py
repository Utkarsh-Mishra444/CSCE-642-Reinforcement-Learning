import asyncio
import os
from seeact.agent import SeeActAgent

# Setup your API Key here, or pass through environment
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key-here"

async def run_agent():
    agent = SeeActAgent(
        model="gpt-4o",      # Main LLM for action prediction
        #model="gpt-5-nano",
        use_vgap_cropping=True,               # Enable VGAP cropping
        vgap_model="ollama/vgap-2k-v1:latest", # VGAP model for cropping
        #vgap_model="ollama/gemma3:4b-it-qat"
        default_task="Find brown leather shoes",
        default_website="https://www.amazon.in/ref=nav_logo"
    )
    await agent.start()
    while not agent.complete_flag:
        prediction_dict = await agent.predict()
        await agent.execute(prediction_dict)
    await agent.stop()

if __name__ == "__main__":
    asyncio.run(run_agent())