# ラウンドロビンで複数のエージェントが大喜利に回答し、ユーザーが面白いと思ったものを選ぶ

import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():

  # Create an OpenAI model client.
  model_client = OpenAIChatCompletionClient(
      model="gemini-2.5-flash",
      api_key="xxx...",
  )

  # 大喜利の回答者エージェントを作成します。
  yamada = AssistantAgent(
      "yamada",
      model_client=model_client,
      system_message="あなたは大喜利の回答者です。面白い回答を考えてください。",
  )

  # 大喜利の回答者エージェントをもう一人作成します。
  tanaka = AssistantAgent(
      "tanaka",
      model_client=model_client,
      system_message="あなたは大喜利の回答者です。面白い回答を考えてください。",
  )

  # 大喜利の回答者エージェントをもう一人作成します。
  katou = AssistantAgent(
      "katou",
      model_client=model_client,
      system_message="あなたは大喜利の回答者です。面白い回答を考えてください。",
  )

  # ユーザーの入力を受け付けるエージェントを作成します。
  user_agent = UserProxyAgent(
      "user",
      input_func=input
  )

  # Define a termination condition that stops the task if the critic approves.
  text_mention_termination = TextMentionTermination("いいじゃん")
  max_messages_termination = MaxMessageTermination(max_messages=10)
  termination = text_mention_termination | max_messages_termination

  # Create a team with the primary and critic agents.
  team = RoundRobinGroupChat([yamada, tanaka, katou, user_agent], termination_condition=termination)

  async for result in team.run_stream(task="猫が副業でやってる仕事ランキング 第1位は？"):
      if isinstance(result, TaskResult):
          print("=== CONVERSATION TERMINATED ===")
      else:
          print(f"{result.source}: {result.content}")
          print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())