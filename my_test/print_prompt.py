from typing import Dict, List
import json

from hipporag.prompts.prompt_template_manager import PromptTemplateManager

prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

passage = "<passage text>"

def print_list_prompt(list_prompt: List[Dict[str, str]]) -> None:
    for message in list_prompt:
        role = message.get("role", "")
        content = message.get("content", "")
        print(f"{role}: {content}")


prompt_type = "ner"
prompt = prompt_template_manager.render(name=prompt_type, passage=passage)
print(f"--- {prompt_type} prompt ---")
print_list_prompt(prompt)
print()

prompt_type = "triple_extraction"
prompt = prompt_template_manager.render(
    name=prompt_type, 
    passage=passage, 
    named_entity_json=json.dumps({"named_entities": ["Entity1", "Entity2"]}))
print(f"--- {prompt_type} prompt ---")
print_list_prompt(prompt)
print()