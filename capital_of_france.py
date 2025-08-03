# フランスの首都を訪ねるコード.

import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
        api_key="xxx...",
    )

    response = await model_client.create([UserMessage(content="What is the capital of France?", source="user")])
    print(response)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())