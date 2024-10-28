from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation
from audio_assistant import AudioAssistant
from utils.updata_realtime_images import ImageSaver
from utils.action_utils import process_robot_actions
import json
# import rospy
import threading

def single_multimodal_call(model_name, base_prompt, query, prompt_img_path, log=True, return_response=True):
    """
    Single round multimodal conversation call with CuriGPT.

    Parameters:
        base_prompt (list): The base prompt for multimodal reasoning.
        query (str): The user query.
        rgb_img_path (str): The path to the real-time image.
        log (bool): Whether to log the response.
        return_response (bool): Whether to return the response.
    """
    # make a copy of the base prompt to create a new prompt
    new_prompt = base_prompt.copy()
    prompt_img_path = prompt_img_path
    new_prompt.append({"role": "user", "content": [{"image": prompt_img_path}, {"text": query}]})

    # check if the model name belongs to qwen-vl-max or qwen-vl-chat-v1
    if model_name not in ['qwen-vl-max', 'qwen-vl-chat-v1']:
        raise ValueError("Model name must be either 'qwen-vl-max' or 'qwen-vl-chat-v1'.")
    # get the response from qwen_vl_max
    if model_name == 'qwen-vl-max':
        response_dict = dashscope.MultiModalConversation.call(
            api_key="sk-06def30695bf49039ed32588306fd25f",
            model="qwen-vl-max-latest",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=new_prompt,
        )


        # response_dict = dashscope.MultiModalConversation.call(model='qwen-vl-max-0809',
        #                                              messages=new_prompt)
        if response_dict.status_code == HTTPStatus.OK:
            # response = response_dict[
            #     "output"]["choices"][0]["message"]["content"]
            response = response_dict.output.choices[0].message.content[0]["text"]
            print("CURI response:\n", response)
        else:
            print(response_dict.code)  # The error code.
            print(response_dict.message)  # The error message.
    # get the response from qwen-vl-chat-v1
    elif model_name == 'qwen-vl-chat-v1':
        dashscope.api_key ="sk-06def30695bf49039ed32588306fd25f"
        response_dict = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                                                                               messages=new_prompt, top_p=0.9, top_k=100)
        if response_dict.status_code == HTTPStatus.OK:
            response = response_dict[
                "output"]["choices"][0]["message"]["content"]
            print("CURI response:\n", response)
        else:
            print(response_dict.code)  # The error code.
            print(response_dict.message)  # The error message.
    if return_response:
        return response

def get_curi_response_with_audio(model_name, api_key, base_url, user_input, curigpt_output, rgb_img_path,
                                 depth_img_path,
                                 local_img_path,
                                 base_multimodal_prompt, rounds=10, realtime_flag=True, prompt_append=False):
    """
    Get CURI response with audio input and output.

    Parameters:
        api_key (str): The OpenAI API key.
        base_url (str): The base URL for the OpenAI API.
        user_input (str): The path to the user audio input file.
        curigpt_output (str): The path to the CuriGPT audio output file.
        rgb_img_path (str): The path to the real-time image.
        depth_img_path (str): The path to the depth image.
        local_img_path (str): The path to the local image.
        base_multimodal_prompt (list): The base prompt for multimodal reasoning.
        rounds (int): The number of rounds.
        realtime_flag (bool): Whether to enable interactive reasoning in real-time.
        prompt_append (bool): Whether to append the model's current response to the next prompt.
    """
    # Create an instance of the AudioAssistant class
    assistant = AudioAssistant(api_key, base_url, user_input, curigpt_output)
    # check if the CuriGPT need to work in the real-time mode
    prompt_img_path = rgb_img_path if realtime_flag else local_img_path

    try:
        for i in range(rounds):
            # save_images_gemini()
            assistant.record_audio()
            transcription = assistant.transcribe_audio()
            response = single_multimodal_call(model_name, base_multimodal_prompt, transcription,
                                              prompt_img_path,
                                              log=True, return_response=True)
            try:
                output_data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return

            if output_data['robot_response']:
                verbal_response = output_data['robot_response']
                assistant.text_to_speech(verbal_response)
                print("CURI audio response:\n", verbal_response)

            if output_data['robot_actions']:
                action_response = output_data['robot_actions']
                print("CURI action response:\n", action_response)
                return process_robot_actions(action_response)

    finally:
        # Ensure all threads are cleaned up properly
        print("Processing complete.")

def process_robot_actions(action_response):
    # Map each action to its corresponding function
    action_map = {
        "tidying_up_bowl": "./ckpt/act/tidying_up_bowl",
        "tidying_up_coaster": "./ckpt/act/tidying_up_coaster",
    }

    if action_response[0]["action"] in action_map:
        if action_response[0]["action"] == "tidying_up_bowl":
            return action_map["tidying_up_bowl"]
        else:
            return action_map["tidying_up_coaster"]
    else:
        print("Unknown action:", action_response['robot_actions'])