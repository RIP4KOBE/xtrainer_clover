"""
CuriGPT
===========================
High-level task reasoning for CURI manipulation via Multimodal Large Language Models (Qwen-VL-Max).
"""

from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation
from curigpt_ros.models.audio_assistant import AudioAssistant
from curigpt_ros.utils.vis_utils import save_color_image_rs, save_images_gemini, plot_image_with_bbox, get_spatial_coordinates
from curigpt_ros.utils.action_utils import process_robot_actions
import json
import rospy
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
        response_dict = dashscope.MultiModalConversation.call(model='qwen-vl-max',
                                                     messages=new_prompt)

        if response_dict.status_code == HTTPStatus.OK:
            response = response_dict[
                "output"]["choices"][0]["message"]["content"][0]["text"]
            print("CURI response:\n", response)

        else:
            print(response_dict.code)  # The error code.
            print(response_dict.message)  # The error message.

    # get the response from qwen-vl-chat-v1
    elif model_name == 'qwen-vl-chat-v1':
        response_dict = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                                                                               messages=new_prompt, top_p=0.9, top_k=100)

        if response_dict.status_code == HTTPStatus.OK:
            response = response_dict[
                "output"]["choices"][0]["message"]["content"]
            print("CURI response:\n", response)

        else:
            print(response_dict.code)  # The error code.
            print(response_dict.message)  # The error message.


    # plot the image with bounding boxes if the response contains actions.
    # plot_image_with_bbox(prompt_img_path, response)

    if return_response:
        return response


def multiple_multimodal_call(base_prompt, query, prompt_img_path, rounds=10, log=True, return_response=False):
    """
    Multiple rounds of  multimodal conversation call with CuriGPT.

    Parameters:
        base_prompt (list): The base prompt for multimodal reasoning.
        query (str): The user query.
        rounds (int): The number of multiple rounds.
        log (bool): Whether to log the response.
        return_response (bool): Whether to return the response.
    """

    new_prompt = base_prompt.copy()
    prompt_img_path = prompt_img_path

    for i in range(rounds):
        new_prompt.append({"role": "user", "content": [{"image": prompt_img_path}, {"text": query}]})

        # call with qwen-vl-max model
        # response_dict = dashscope.MultiModalConversation.call(model='qwen-vl-max',
        #                                                  messages=new_prompt)

        # call with qwen-vl-chat-v1 model
        response_dict = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                                    messages=new_prompt, top_p=0.9, top_k=100)

        # call with qwen-vl-plus model
        # response_dict = MultiModalConversation.call(model='qwen-vl-plus',
        #                                            messages=new_prompt, top_p=0.1, top_k=10)

        response = response_dict[
            "output"]['choices'][0]['message']['content']

        if response_dict.status_code == HTTPStatus.OK:
            print("CURI response:\n", response)
        else:
            print(response_dict.code)  # The error code.
            print(response_dict.message)  # The error message.

        if return_response:
            return response


def get_curi_response(base_multimodal_prompt, rounds=10, prompt_append=False):

    """Get CURI response for a given number of rounds."""
    inference_results = None
    if not prompt_append:
        print("without prompt append")
        for i in range(rounds):
            instruction = input("Please input your instruction: ")
            inference_results = single_multimodal_call(base_multimodal_prompt, instruction, log=True, return_response=False)

    else:
        if rounds < 1:
            raise ValueError("Number of rounds must be at least 1.")

        for i in range(rounds):
            instruction = input("Please input your instruction: ")

            # Single call for each round
            if i == 1:
                # For the first round or if not returning response, just update the prompt
                inference_results = single_multimodal_call(base_multimodal_prompt, instruction, log=True,
                                                           return_response=False)

            else:
                # For subsequent rounds, get the response as well to append to the prompt
                new_prompt, response_dict = single_multimodal_call(base_multimodal_prompt, instruction, log=True,
                                                                   return_response=True)
                if 'output' in response_dict and 'choices' in response_dict['output'] and len(
                        response_dict['output']['choices']) > 0:
                    # Append the model's response to the prompt for the next round
                    response_content = response_dict['output']['choices'][0]['message']['content']
                    response_role = response_dict['output']['choices'][0]['message']['role']
                    new_prompt.append({'role': response_role, 'content': [{'text': response_content}]})
                base_multimodal_prompt = new_prompt


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

    rospy.init_node('curigpt_response', anonymous=True)

    # Create an instance of the AudioAssistant class
    assistant = AudioAssistant(api_key, base_url, user_input, curigpt_output)

    # check if the CuriGPT need to work in the real-time mode
    prompt_img_path = rgb_img_path if realtime_flag else local_img_path

    # Start the image saving thread
    image_thread = threading.Thread(target=save_images_gemini)
    image_thread.start()

    if not prompt_append:
        try:
            for i in range(rounds):
                # save_images_gemini()
                assistant.record_audio()
                transcription = assistant.transcribe_audio()

                response = single_multimodal_call(model_name, base_multimodal_prompt, transcription,
                                                  prompt_img_path,
                                                  log=True, return_response=True)

                # play the audio response
                if response is not None:
                    assistant.text_to_speech(response)

                # parse the JSON string of response.
                # try:
                #     output_data = json.loads(response)
                # except json.JSONDecodeError as e:
                #     print(f"Error decoding JSON: {e}")
                #     return
                #
                # if output_data['robot_response']:
                #     verbal_response = output_data['robot_response']
                #     assistant.text_to_speech(verbal_response)
                #     print("CURI audio response:\n", verbal_response)
                #
                # if output_data['robot_actions']:
                #     action_response = output_data['robot_actions']
                #     print("CURI action response:\n", action_response)
                #     process_robot_actions(action_response, rgb_img_path, depth_img_path)


        finally:
            # Ensure all threads are cleaned up properly
            rospy.signal_shutdown("Shutting down ROS node.")
            image_thread.join()  # Wait for the image saving thread to finish
            print("Processing complete.")

    else:
        if rounds < 1:
            raise ValueError("Number of rounds must be at least 1.")

        for i in range(rounds):
            instruction = input("Please input your instruction: ")

            # Single call for each round
            if i == 1:
                # for the first round or if not returning response, just update the prompt
                inference_results = single_multimodal_call(base_multimodal_prompt, instruction, log=True,
                                                           return_response=False)

            else:
                # For subsequent rounds, get the response as well to append to the prompt
                new_prompt, response_dict = single_multimodal_call(base_multimodal_prompt, instruction, log=True,
                                                                   return_response=True)
                if 'output' in response_dict and 'choices' in response_dict['output'] and len(
                        response_dict['output']['choices']) > 0:
                    # Append the model's response to the prompt for the next round
                    response_content = response_dict['output']['choices'][0]['message']['content']
                    response_role = response_dict['output']['choices'][0]['message']['role']
                    new_prompt.append({'role': response_role, 'content': [{'text': response_content}]})
                base_multimodal_prompt = new_prompt


if __name__ == '__main__':

    # Load the configuration from the config.json file
    with open('config/config.json', 'r') as config_file:
        config = json.load(config_file)

    # accessing configuration variables
    api_key = config['openai_api_key']
    base_url = config['base_url']
    user_input_filename = config['user_input_filename']
    curigpt_output_filename = config['curigpt_output_filename']
    depth_img_path = config['depth_img_path']
    rgb_img_path = config['rgb_img_path']
    local_img_path = config['local_img_path']
    local_img_path_chinese = "assets/img/curigpt_demo_huawei.png"
    model_name = config['model_name']

    base_multimodal_prompt = [
        # Example 1: Speech-to-Speech Reasoning
        {
            "role": "user",
            "content": [
                {"image": local_img_path_chinese},
                {"text": "下午好, 你能描述下你现在看到的场景吗?"},
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "text": "好的，我看到一个白色的桌子上放着一些物品。左上角有一个橙色的洗涤剂，中间有一个绿色的绿茶罐子，一个蓝色的午餐肉罐头和一个红色的盘子，盘子里放着一根香蕉。"}]
        },
        {
            "role": "user",
            "content": [
                {"image": local_img_path_chinese},
                {"text": "你面前的桌子上是否呢？"},
            ]
        },
        {
            "role": "assistant",
            "content": [{
                            "text": "我认为午餐肉罐头适合作为吃火锅时的配菜，因为它易于烹饪，可以在火锅中快速煮熟，并可以与其他食材搭配，增加火锅的多样性"}]
        },
        {
            "role": "user",
            "content": [
                {"image": local_img_path_chinese},
                {"text": "我想清洗一下红色盘子，你可以把洗涤剂递给我吗"},
            ]
        },
        {
            "role": "assistant",
            "content": [{"text": "好的，没问题。我将把洗涤剂瓶子递给你。"}]
        },
    ]

    get_curi_response_with_audio(model_name, api_key, base_url, user_input_filename, curigpt_output_filename,
                                 rgb_img_path,
                                 depth_img_path, local_img_path, base_multimodal_prompt, rounds=10,
                                 realtime_flag=True, prompt_append=False)