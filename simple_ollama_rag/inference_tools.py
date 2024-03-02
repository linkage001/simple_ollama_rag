import ollama
from .tools import format_docs, decorator_timer

hashtags_prompt = """{
"What's the best way to learn a new language?": [
    "#LanguageLearning", "#PolyglotLife", "#SpeakUp", "#BilingualGoals", "#CultureConnect"
],
"How can I improve my productivity at work?": [
    "#ProductivityHacks", "#TimeManagement", "#WorkSmart", "#EfficiencyBoost", "#TaskMaster"
],
"What are some healthy breakfast options?": [
    "#HealthyEating", "#MorningFuel", "#NutritionTips", "#WellnessChoices", "#BreakfastIdeas"
],
"What's the most effective workout for building muscle?": [
    "#StrengthTraining", "#MuscleGains", "#FitnessGoals", "#LiftHeavy", "#SweatItOut"
],
"How do I stay motivated when pursuing long-term goals?": [
    "#GoalSetting", "#PersistencePays", "#DreamBig", "#StayFocused", "#NeverGiveUp"
],
"$$QUESTION$$": [
    "#"""

def ollama_chat(question, context, inference_model, inference_config, system=None):

    chat_ml_system = f"""<|im_start|>system
{system}<|im_end|>"""
    chat_ml = f"""
<|im_start|>context
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

    formatted_prompt = chat_ml
    if system is not None:
        formatted_prompt = chat_ml_system + chat_ml
    response = ollama.chat(model=inference_model, messages=[{'role': 'user', 'content': formatted_prompt}],
                           options=inference_config)
    return formatted_prompt, response

@decorator_timer
def rag_chain(question, retriever, inference_model, inference_config, system, create_hashtags):
    if create_hashtags:
        hashtags = ollama.chat(model=inference_model, messages=[
            {'role': 'user', 'content': hashtags_prompt.replace('$$QUESTION$$', question)}],
                               options={"stop": [
                                   "\n",
                                   "]"
                               ]})["message"]["content"]
        question = f'{question} "#"{hashtags}'
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_chat(question, formatted_context, inference_model, inference_config, system)
