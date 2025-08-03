# Selectorグループチャット形式で、スーパーマーケットの客にサービスを提供するチームの例.

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():

  # Create an OpenAI model client.
  model_client = OpenAIChatCompletionClient(
      model="gemini-2.5-flash",
      api_key="xxx...",
  )

  # 献立プランナーエージェント.
  dish_planner_agent = AssistantAgent(
      "dish_planner",
      description="スーパーの献立プランナー。お客さんが作りたい料理を提案します。提案が終わったら、'FINISH'と返答します。",
      model_client=model_client,
      system_message="あなたはスーパーの献立プランナーです。予算を考えずに、お客さんの気分にぴったりな料理を一品考えます。提案が終わったら、'FINISH'と返答してください。",
  )

  # 商品選びエージェント.
  ingredient_coordinator_agent = AssistantAgent(
      "ingredient_coordinator",
      description="スーパーの食材コーディネーター。献立プランナーが考えた料理に必要な食材を考えます。提案が終わったら、'FINISH'と返答します。",
      model_client=model_client,
      system_message="あなたはスーパーの食材コーディネーターです。献立プランナーが考えた料理に必要な食材を提案します。食材を選ぶ際は、季節や地域の特産品を考慮してください。それぞれの食材の個数を言ってください。提案が終わったら、'FINISH'と返答してください。",
  )

  # 清算担当エージェント.
  cashier_agent = AssistantAgent(
      "cashier",
      description="スーパーの清算担当。選び終わった食材を精算します。合計金額がお客さんの予算内におさまったら、金額を伝え、'TERMINATE'と返答します。予算を超えた場合は、予算内に収まるように食材を減らすようにingredient_coordinatorに提案します。提案が終わったら、'REJECTED'と返答します。",
      model_client=model_client,
      system_message="あなたはスーパーの清算担当です。商品選びエージェントが提案した食材のそれぞれの価格を現実的に妥当な範囲で考えて、合計金額を計算します。合計金額がお客さんの予算内におさまったら、金額を伝え、'TERMINATE'と返答してください。予算を超えた場合は、予算内に収まるように食材を減らすように商品選びエージェントに提案してください。提案が終わったら、'FINISH'と返答してください。",
  )

  text_mention_termination = TextMentionTermination("TERMINATE")
  max_messages_termination = MaxMessageTermination(max_messages=10)
  termination = text_mention_termination | max_messages_termination

  selector_prompt = """あなたはスーパーマーケットのエージェントチームを指揮するAIです。
  以下のエージェントがいます:
    {roles}

    これまでの会話履歴:
    {history}

    上記の会話を読んで、次に発言するエージェントを {participants} から選択してください。
    1人のエージェントのみを選択してください。
    """


  team = SelectorGroupChat(
    [dish_planner_agent, ingredient_coordinator_agent, cashier_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
  )
  result = await team.run(task="なんだかステーキを使ったガッツリしたものを食べたいです。予算は500円です。")
  
  print("=== CONVERSATION MESSAGES ===")
  for i, message in enumerate(result.messages, 1):
      print(f"\n{i}. {message.source.upper()}:")
      print(f"   {message.content}")
  
  print(f"\n=== STOP REASON ===")
  print(f"   {result.stop_reason}")

if __name__ == "__main__":
    asyncio.run(main())