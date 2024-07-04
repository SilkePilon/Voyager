import re
import time
import ollama
import ast

import voyager.utils as U
from javascript import require
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from voyager.prompts import load_prompt
from voyager.control_primitives_context import load_control_primitives_context


class ActionAgent:

    def __init__(
        self,
        model_name="llama2",
        temperature=0,
        request_timout=120,
        ckpt_dir="ckpt",
        resume=False,
        chat_log=True,
        execution_error=True,
    ):
        self.ckpt_dir = ckpt_dir
        self.chat_log = chat_log
        self.execution_error = execution_error
        self.model_name = model_name
        self.temperature = temperature
        U.f_mkdir(f"{ckpt_dir}/action")
        if resume:
            print(f"\033[32mLoading Action Agent from {
                  ckpt_dir}/action\033[0m")
            self.chest_memory = U.load_json(
                f"{ckpt_dir}/action/chest_memory.json")
        else:
            self.chest_memory = {}

        def chat(messages):
            formatted_messages = []
            for m in messages:
                if isinstance(m, SystemMessage):
                    formatted_messages.append(
                        {"role": "system", "content": m.content})
                elif isinstance(m, HumanMessage):
                    formatted_messages.append(
                        {"role": "user", "content": m.content})
                else:
                    print(f"\033[32mUnknown message type: {type(m)}\033[0m")
            response = ollama.chat(
                model=model_name,
                messages=formatted_messages,
            )
            return AIMessage(content=response['message']['content'])

        # def chat(messages):
        #     response = ollama.chat(model=self.model_name, messages=messages)
        #     return AIMessage(content=response['message']['content'])

        self.llm = chat

    def update_chest_memory(self, chests):
        for position, chest in chests.items():
            if position in self.chest_memory:
                if isinstance(chest, dict):
                    self.chest_memory[position] = chest
                if chest == "Invalid":
                    print(
                        f"\033[32mAction Agent removing chest {
                            position}: {chest}\033[0m"
                    )
                    self.chest_memory.pop(position)
            else:
                if chest != "Invalid":
                    print(f"\033[32mAction Agent saving chest {
                          position}: {chest}\033[0m")
                    self.chest_memory[position] = chest
        U.dump_json(self.chest_memory, f"{
                    self.ckpt_dir}/action/chest_memory.json")

    def render_chest_observation(self):
        chests = []
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) > 0:
                chests.append(f"{chest_position}: {chest}")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, dict) and len(chest) == 0:
                chests.append(f"{chest_position}: Empty")
        for chest_position, chest in self.chest_memory.items():
            if isinstance(chest, str):
                assert chest == "Unknown"
                chests.append(f"{chest_position}: Unknown items inside")
        assert len(chests) == len(self.chest_memory)
        if chests:
            chests = "\n".join(chests)
            return f"Chests:\n{chests}\n\n"
        else:
            return f"Chests: None\n\n"

    def render_system_message(self, skills=[]):
        system_template = load_prompt("action_template")
        # FIXME: Hardcoded control_primitives
        base_skills = [
            "exploreUntil",
            "mineBlock",
            "craftItem",
            "placeItem",
            "smeltItem",
            "killMob",
        ]
        if self.model_name != "llama2":
            base_skills += [
                "useChest",
                "mineflayer",
            ]
        programs = "\n\n".join(
            load_control_primitives_context(base_skills) + skills)
        response_format = load_prompt("action_response_format")
        system_message = SystemMessage(content=system_template.format(
            programs=programs, response_format=response_format
        ))
        return system_message

    def render_human_message(
        self, *, events, code="", task="", context="", critique=""
    ):
        chat_messages = []
        error_messages = []
        # FIXME: damage_messages is not used
        damage_messages = []
        assert events[-1][0] == "observe", "Last event must be observe"
        for i, (event_type, event) in enumerate(events):
            if event_type == "onChat":
                chat_messages.append(event["onChat"])
            elif event_type == "onError":
                error_messages.append(event["onError"])
            elif event_type == "onDamage":
                damage_messages.append(event["onDamage"])
            elif event_type == "observe":
                biome = event["status"]["biome"]
                time_of_day = event["status"]["timeOfDay"]
                voxels = event["voxels"]
                entities = event["status"]["entities"]
                health = event["status"]["health"]
                hunger = event["status"]["food"]
                position = event["status"]["position"]
                equipment = event["status"]["equipment"]
                inventory_used = event["status"]["inventoryUsed"]
                inventory = event["inventory"]
                assert i == len(events) - 1, "observe must be the last event"

        observation = ""

        if code:
            observation += f"Code from the last round:\n{code}\n\n"
        else:
            observation += f"Code from the last round: No code in the first round\n\n"

        if self.execution_error:
            if error_messages:
                error = "\n".join(error_messages)
                observation += f"Execution error:\n{error}\n\n"
            else:
                observation += f"Execution error: No error\n\n"

        if self.chat_log:
            if chat_messages:
                chat_log = "\n".join(chat_messages)
                observation += f"Chat log: {chat_log}\n\n"
            else:
                observation += f"Chat log: None\n\n"

        observation += f"Biome: {biome}\n\n"

        observation += f"Time: {time_of_day}\n\n"

        if voxels:
            observation += f"Nearby blocks: {', '.join(voxels)}\n\n"
        else:
            observation += f"Nearby blocks: None\n\n"

        if entities:
            nearby_entities = [
                k for k, v in sorted(entities.items(), key=lambda x: x[1])
            ]
            observation += f"Nearby entities (nearest to farthest): {
                ', '.join(nearby_entities)}\n\n"
        else:
            observation += f"Nearby entities (nearest to farthest): None\n\n"

        observation += f"Health: {health:.1f}/20\n\n"

        observation += f"Hunger: {hunger:.1f}/20\n\n"

        observation += f"Position: x={position['x']:.1f}, y={
            position['y']:.1f}, z={position['z']:.1f}\n\n"

        observation += f"Equipment: {equipment}\n\n"

        if inventory:
            observation += f"Inventory ({inventory_used}/36): {inventory}\n\n"
        else:
            observation += f"Inventory ({inventory_used}/36): Empty\n\n"

        if not (
            task == "Place and deposit useless items into a chest"
            or task.startswith("Deposit useless items into the chest at")
        ):
            observation += self.render_chest_observation()

        observation += f"Task: {task}\n\n"

        if context:
            observation += f"Context: {context}\n\n"
        else:
            observation += f"Context: None\n\n"

        if critique:
            observation += f"Critique: {critique}\n\n"
        else:
            observation += f"Critique: None\n\n"

        return HumanMessage(content=observation)

    def process_ai_message(self, message):
        print("Processing AI message")
        assert isinstance(message, AIMessage)

        retry = 3
        error = None
        while retry > 0:
            print(f"Retry: {retry}")
            try:
                # Extract code from markdown code blocks
                code_pattern = re.compile(
                    r"```(?:javascript|js)(.*?)```", re.DOTALL)
                code = "\n".join(code_pattern.findall(message.content))

                if not code:
                    return "No JavaScript code found in the message."

                # Parse the code to find functions
                functions = []
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        is_async = isinstance(node, ast.AsyncFunctionDef)
                        functions.append({
                            "name": node.name,
                            "type": "AsyncFunctionDeclaration" if is_async else "FunctionDeclaration",
                            "body": ast.get_source_segment(code, node),
                            "params": [arg.arg for arg in node.args.args]
                        })

                if not functions:
                    return "No functions found in the JavaScript code."

                # Find the main function (last async function or last function if no async)
                async_functions = [
                    f for f in functions if f["type"] == "AsyncFunctionDeclaration"]
                main_function = async_functions[-1] if async_functions else functions[-1]

                # Validate main function
                if len(main_function["params"]) != 1 or main_function["params"][0] != "bot":
                    return f"Main function {main_function['name']} must take a single argument named 'bot'"

                # Prepare the code for execution
                program_code = "\n\n".join(
                    function["body"] for function in functions)
                exec_code = f"await {main_function['name']}(bot);"

                return {
                    "program_code": program_code,
                    "program_name": main_function["name"],
                    "exec_code": exec_code,
                }

            except Exception as e:
                print(f"Error parsing action response: {e}")
                retry -= 1
                error = e
                time.sleep(1)

        return f"Error parsing action response (before program execution): {error}"

    def summarize_chatlog(self, events):
        def filter_item(message: str):
            craft_pattern = r"I cannot make \w+ because I need: (.*)"
            craft_pattern2 = (
                r"I cannot make \w+ because there is no crafting table nearby"
            )
            mine_pattern = r"I need at least a (.*) to mine \w+!"
            if re.match(craft_pattern, message):
                return re.match(craft_pattern, message).groups()[0]
            elif re.match(craft_pattern2, message):
                return "a nearby crafting table"
            elif re.match(mine_pattern, message):
                return re.match(mine_pattern, message).groups()[0]
            else:
                return ""

        chatlog = set()
        for event_type, event in events:
            if event_type == "onChat":
                item = filter_item(event["onChat"])
                if item:
                    chatlog.add(item)
        return "I also need " + ", ".join(chatlog) + "." if chatlog else ""
