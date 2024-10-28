"""
CuriGPT
===========================
High-level task reasoning for CURI manipulation via Multimodal Large Language Models (Qwen-VL-Max).
"""

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

    # rospy.init_node('curigpt_response', anonymous=True)

    # Create an instance of the AudioAssistant class
    assistant = AudioAssistant(api_key, base_url, user_input, curigpt_output)

    # check if the CuriGPT need to work in the real-time mode
    prompt_img_path = rgb_img_path if realtime_flag else local_img_path

    if not prompt_append:
        try:
            for i in range(rounds):
                # save_images_gemini()
                assistant.record_audio()
                transcription = assistant.transcribe_audio()

                response = single_multimodal_call(model_name, base_multimodal_prompt, transcription,
                                                  prompt_img_path,
                                                  log=True, return_response=True)

                # print("Type of response:", type(response))
                # print("Response content:", response)
                # parse the JSON string of response.
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
                    process_robot_actions(action_response, rgb_img_path, depth_img_path)


        finally:
            # Ensure all threads are cleaned up properly
            image_saver.rgb_img_sub.join()
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
    with open('../config/config.json', 'r') as config_file:
        config = json.load(config_file)

    # accessing configuration variables
    api_key = config['openai_api_key']
    base_url = config['base_url']
    user_input_filename = config['user_input_filename']
    curigpt_output_filename = config['curigpt_output_filename']
    depth_img_path = config['depth_img_path']
    rgb_img_path = config['rgb_img_path']
    local_img_path = config['local_img_path']
    model_name = config['model_name']

    base_multimodal_prompt = [
        {
            "role": "system",
            "content": [{
                "text": '''You are an excellent responser of human instructions for household tabletop tasks. Given an verbal instruction and an image of the tabletop scenario, you respond to the human instruction and select appropriate robot actions if necessary.

                        Your response must be output in a structured JSON format and contain the following two keywords:
                        - "robot_response" for the verbal response to the human instruction.
                        - "robot_actions" for the description of the physical action you will perform, including the specific action name and parameters for bounding box coordinates.
                        Must add the "," delimiter between the two keywords to ensure proper JSON formatting.

                        Two robot actions are available to you:
                        - grasp_and_place(arg1, arg2): The robot grasps the object located at the bounding box coordinates specified by `arg1`, and places it at the location specified by `arg2`.
                        - grasp_and_give(arg1): The robot picks up the object identified by the bounding box coordinates `arg1` and hands it directly to the user. This action requires only arg1 for the coordinates of the object to be picked up.
                        Note that both arg1 and arg2 should be detected by yourself from the provided image.'''
            }]
        },
        {
            "role": "user",
            "content": [
                {"image": local_img_path},
                {"text": "hey CURI, what do you see right now?"},
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "text": json.dumps({
                    "robot_response": "Now I see a red plate in the center of the table, and an empty spam can, a banana, and a soda can near the plate. There is also a green container on the upper left corner of the table.",
                    "robot_actions": None
                }, indent=4)
            }]
        },
        {
            "role": "user",
            "content": [
                {"image": local_img_path},
                {"text": "Can you give me the soda can on the table?"}
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "text": json.dumps({
                    "robot_response": "Sure, here is your soda can.",
                    "robot_actions": [
                        {
                            "action": "grasp_and_give",
                            "parameters": {
                                "arg1": {
                                    "description": "soda can",
                                    "bbox_coordinates": [634, 672, 815, 780] # Hypothetical bbox coordinates [x1, y1, x2, y2] for the soda can
                                }
                            }
                        }
                    ]
                }, indent=4)
            }]
        },
        {
            "role": "user",
            "content": [
                {"image": local_img_path},
                {"text": "Can you put the spam can in the container?"}
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "text": json.dumps({
                    "robot_response": "Sure thing.",
                    "robot_actions": [
                        {
                            "action": "grasp_and_place",
                            "parameters": {
                                "arg1": {
                                    "description": "spam can",
                                    "bbox_coordinates": [139, 719, 317, 862]  # Hypothetical bbox coordinates [x1, y1, x2, y2] for the spam can
                                },
                                "arg2": {
                                    "description": "container",
                                    "bbox_coordinates": [579, 67, 961, 300]  # Hypothetical bbox coordinates [x1, y1, x2, y2] for the container
                                }
                            }
                        }
                    ]
                }, indent=4)
            }]
        }
    ]

    # get the CURI response with audio input and output

    # base_multimodal_prompt = [
    #     {
    #         "role": "system",
    #         "content": [{
    #             "text": '''You are an excellent responser of human instructions for household tabletop tasks. Given an verbal instruction and an image of the tabletop scenario, you respond to the human instruction and select appropriate robot actions if necessary.
    #
    #                     Your response must be output in a structured JSON format and contain the following two keywords:
    #                     - "robot_response" for the verbal response to the human instruction.
    #                     - "robot_actions" for the description of the physical action you will perform, including the specific action name and parameters for bounding box coordinates.
    #                     Must add the "," delimiter between the two keywords to ensure proper JSON formatting.
    #
    #                     Two robot actions are available to you:
    #                     - grasp_and_place(arg1, arg2): The robot grasps the object located at the bounding box coordinates specified by `arg1`, and places it at the location specified by `arg2`.
    #                     - grasp_and_give(arg1): The robot picks up the object identified by the bounding box coordinates `arg1` and hands it directly to the user. This action requires only arg1 for the coordinates of the object to be picked up.
    #                     Note that both arg1 and arg2 should be detected by yourself from the provided image.'''
    #         }]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": local_img_path},
    #             {"text": "hey CURI, what do you see right now?"},
    #         ]
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [{
    #             "text": json.dumps({
    #                 "robot_response": "Now I see a white table with various objects on it, including a banana, an apple, a spray bottle, a green box, and a red game controller.",
    #                 "robot_actions": None
    #             }, indent=4)
    #         }]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": local_img_path},
    #             {"text": "do you know how to use the spray bottle?"},
    #         ]
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [{
    #             "text": json.dumps({
    #                 "robot_response": "Yes, I know how to use a spray bottle. It's a common household item used for cleaning or other purposes. To use it, you typically press down on the top part of the bottle, which releases the liquid inside through a nozzle, creating a fine mist or stream of liquid.",
    #                 "robot_actions": None
    #             }, indent=4)
    #         }]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": local_img_path},
    #             {"text": "Can you handover the spray bottle to me?"}
    #         ]
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [{
    #             "text": json.dumps({
    #                 "robot_response": "Sure thing, here is the spray bottle.",
    #                 "robot_actions": [
    #                     {
    #                         "action": "grasp_and_give",
    #                         "parameters": {
    #                             "arg1": {
    #                                 "description": "spray bottle",
    #                                 "bbox_coordinates": [634, 672, 815, 780]
    #                                 # Hypothetical coordinates for the spray bottle
    #                                 # [x1, y1, x2, y2]
    #                             }
    #                         }
    #                     }
    #                 ]
    #             }, indent=4)
    #         }]
    #     }
    #     # {
    #     #     "role": "user",
    #     #     "content": [
    #     #         {"image": local_img_path},
    #     #         {"text": "Can you put the spam can in the container?"}
    #     #     ]
    #     # },
    #     # {
    #     #     "role": "assistant",
    #     #     "content": [{
    #     #         "text": json.dumps({
    #     #             "robot_response": "Sure thing.",
    #     #             "robot_actions": [
    #     #                 {
    #     #                     "action": "grasp_and_place",
    #     #                     "parameters": {
    #     #                         "arg1": {
    #     #                             "description": "spam can",
    #     #                             "bbox_coordinates": [139, 719, 317, 862]  # [x1, y1, x2, y2]
    #     #                         },
    #     #                         "arg2": {
    #     #                             "description": "container",
    #     #                             "bbox_coordinates": [579, 67, 961, 300]  # [x1, y1, x2, y2]
    #     #                         }
    #     #                     }
    #     #                 }
    #     #             ]
    #     #         }, indent=4)
    #     #     }]
    #     # }
    # ]

    # base_multimodal_prompt = [
    #     {
    #         "role": "system",
    #         "content": [{
    #             "text": '''You are an excellent responser of human instructions for household tabletop tasks. Given an verbal instruction and an image of the tabletop scenario, you respond to the human instruction and select appropriate robot actions if necessary.
    #
    #                     Your response must be output in a structured JSON format and contain the following two keywords:
    #                     - "robot_response" for the verbal response to the human instruction.
    #                     - "robot_actions" for the description of the physical action you will perform, including the specific action name and parameters for bounding box coordinates.
    #                     Must add the "," delimiter between the two keywords to ensure proper JSON formatting.
    #
    #                     Two robot actions are available to you:
    #                     - grasp_and_place(arg1, arg2): The robot grasps the object located at the bounding box coordinates specified by `arg1`, and places it at the location specified by `arg2`.
    #                     - grasp_and_give(arg1): The robot picks up the object identified by the bounding box coordinates `arg1` and hands it directly to the user. This action requires only arg1 for the coordinates of the object to be picked up.
    #                     Note that both arg1 and arg2 should be detected by yourself from the provided image.'''
    #         }]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": local_img_path},
    #             {"text": "hey CURI, what do you see right now?"},
    #         ]
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [{
    #             "text": json.dumps({
    #                 "robot_response": "Now I see a white table with various objects on it, including a banana, an apple, a spray bottle, a green box, and a red game controller.",
    #                 "robot_actions": None
    #             }, indent=4)
    #         }]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": local_img_path},
    #             {"text": "I wan to eat some fruits, can you give me one?"},
    #         ]
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [{
    #             "text": json.dumps({
    #                 "robot_response": "There's a banana and an apple on the table. Which one do you prefer?",
    #                 "robot_actions": None
    #             }, indent=4)
    #         }]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": local_img_path},
    #             {"text": "Give me the apple, please."}
    #         ]
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [{
    #             "text": json.dumps({
    #                 "robot_response": "Sure, Here's the apple for you.",
    #                 "robot_actions": [
    #                     {
    #                         "action": "grasp_and_give",
    #                         "parameters": {
    #                             "arg1": {
    #                                 "description": "apple",
    #                                 "bbox_coordinates": [435, 320, 496, 437]  # Hypothetical coordinates for the apple
    #                             }
    #                         }
    #                     }
    #                 ]
    #             }, indent=4)
    #         }]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {"image": local_img_path},
    #             {"text": "Can you put the banana in the green box?"}
    #         ]
    #     },
    #     {
    #         "role": "assistant",
    #         "content": [{
    #             "text": json.dumps({
    #                 "robot_response": "Sure thing.",
    #                 "robot_actions": [
    #                     {
    #                         "action": "grasp_and_place",
    #                         "parameters": {
    #                             "arg1": {
    #                                 "description": "banana",
    #                                 "bbox_coordinates": [379, 486, 474, 652] # Hypothetical coordinates for the banana
    #
    #                             },
    #                             "arg2": {
    #                                 "description": "green box",
    #                                 "bbox_coordinates": [538, 284, 709, 656]  # Hypothetical coordinates for the green box
    #                             }
    #                         }
    #                     }
    #                 ]
    #             }, indent=4)
    #         }]
    #     }
    # ]


    get_curi_response_with_audio(model_name, api_key, base_url, user_input_filename, curigpt_output_filename,
                                 rgb_img_path,
                                 depth_img_path, local_img_path, base_multimodal_prompt, rounds=10,
                                 realtime_flag=True, prompt_append=False)