# ラウンドロビン方式のグループチャットを使用して、秋に関する詩を作成するエージェントの例.

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():

  # Create an OpenAI model client.
  model_client = OpenAIChatCompletionClient(
      model="gemini-2.5-flash",
      api_key="xxx...",
  )

  # Create the primary agent.
  primary_agent = AssistantAgent(
      "primary",
      model_client=model_client,
      system_message="You are a helpful AI assistant.",
  )

  # Create the critic agent.
  critic_agent = AssistantAgent(
      "critic",
      model_client=model_client,
      system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed. Don't use 'APPROVE' in your feedbacks. Don't write a poem by yourself.",
  )

  # Define a termination condition that stops the task if the critic approves.
  text_termination = TextMentionTermination("APPROVE")

  # Create a team with the primary and critic agents.
  team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

  result = await team.run(task="秋についての詩を作成してください。")
  
  print("=== CONVERSATION MESSAGES ===")
  for i, message in enumerate(result.messages, 1):
      print(f"\n{i}. {message.source.upper()}:")
      print(f"   {message.content}")
  
  print(f"\n=== STOP REASON ===")
  print(f"   {result.stop_reason}")

if __name__ == "__main__":
    asyncio.run(main())