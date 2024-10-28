"""
CuriGPT
===========================

High-level task reasoning for CURI manipulation via Multimodal LLM (Qwen-VL-Max).
"""

from http import HTTPStatus
import dashscope
from dashscope import MultiModalConversation
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from audio_assistant import AudioAssistant

local_file_path1 = '../../assets/img/curigpt_demo_huawei.png'

base_multimodal_prompt = [
#     {
#         "role": "system",
#         "content": [{
#             "text": '''You are a multimodal large language model serving as the brain for a humanoid robot. Your capabilities include understanding and processing both visual data and natural language. Here is what you need to do:
#
# 1. **Speech-to-Speech Reasoning**: When provided with a human query and a scene image, analyze the image, understand the query's context, and generate an appropriate verbal response that demonstrates your understanding of the image content. The robot action is None in this case.
#
# 2. **Speech-to-Action Reasoning**: When the human command involves a task that you should perform, assess the necessary action, identify the relevant object in the image, and determine the bounding box coordinates for that object. Then, formulate a response plan to execute the task.
#
# Upon processing the information, output your responses in a structured JSON format with the following keys:
#
# - "robot_response" for the verbal response to the human query.
# - "robot_actions" for the description of the physical action you will perform, including the bounding box coordinates of the object you will manipulate.''',
#
#             "extra": '''ROBOT ACTION LIST is defined as follows:
# grasp_and_place(arg1, arg2): The robot grasps the object at position arg1 and places it at position arg2.
# grasp_handover_place(arg1, arg2): The robot grasps the object at position arg1 with the left hand, hands it over to the right hand, and places it at position arg2.
# grasp_and_give(arg1): The robot grasps the object at position arg1 and gives it to the user. Bounding box coordinates (arg1, arg2) should be determined by you based on the image provided. Here are some examples of expected inputs and outputs:'''
#         }]
#     },
    # Example 1: Speech-to-Speech Reasoning
    {
        "role": "user",
        "content": [
            {"image": local_file_path1},
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
    # Example 2: Speech-to-Action Reasoning
    {
        "role": "user",
        "content": [
            {"image": local_file_path1},
            {"text": "Can you give me something to drink?"}
        ]
    },
    {
        "role": "assistant",
        "content": [{
            "text": json.dumps({
                "robot_response": "Sure, you can have the soda to drink.",
                "robot_actions": [
                    {
                        "action": "grasp_and_give",
                        "parameters": {
                            "arg1": {
                                "description": "soda can",
                                "bbox_coordinates": [634, 672, 815, 780]  # Hypothetical coordinates for the soda can
                                # [x1, y1, x2, y2]
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
            {"image": local_file_path1},
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
                                    "bbox_coordinates": [139, 719, 317, 862] # [x1, y1, x2, y2]
                                },
                                "arg2": {
                                    "description": "container",
                                    "bbox_coordinates": [579, 67, 961, 300] # [x1, y1, x2, y2]
                                }
                    }
                }
                ]
            }, indent=4)
        }]
    }
]

def add_bbox_patch(ax, draw, bbox, color, w, h, caption):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    # draw.rectangle([x1, y1, x2, y2], outline=color, width=10)

    # Annotate the bounding box with a caption
    ax.text(x1, y1, caption, color=color, weight='bold', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.1'))

def plot_image_with_bbox(image_path, response):
    """Parse the bbox coordinates from the response of mllm and plot them on the image.

    Parameters:
    image_path (str): The path to the image file.
    response (str): The response from the multimodal large language model.
    """

    # Parse the JSON string.
    try:
        output_data = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Load the image.
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    w, h = image.size # get the width and height od the image size


    # Create a plot.
    fig, ax = plt.subplots()

    # Display the image.
    ax.imshow(image)

    # Go through each action and plot the bounding boxes.
    if output_data['robot_actions'] is not None:
        for action in output_data['robot_actions']:
            bbox1 = action['parameters']['arg1']['bbox_coordinates']
            caption1 = action['parameters']['arg1']['description']

            # check if arg2 exists
            if 'arg2' in action['parameters']:
                bbox2 = action['parameters']['arg2']['bbox_coordinates']
                caption2 = action['parameters']['arg2']['description']

                add_bbox_patch(ax, draw, bbox1, 'red', w, h, caption1)
                add_bbox_patch(ax, draw, bbox2, 'blue', w, h, caption2)

            else:
                add_bbox_patch(ax, draw, bbox1, 'red', w, h, caption1)

            # Show the plot with the bounding boxes.
        plt.show()
        # image.show()
    # else:
    #     print("The action is None.")



def single_multimodal_call(base_prompt, query, log=True, return_response=True):

    """single round multimodal conversation call with CURIGPT.
    """

    # Make a copy of the base prompt to avoid modifying the original
    new_prompt = base_prompt.copy()
    new_prompt.append({"role": "user", "content": [{"image": local_file_path1}, {"text": query}]})
    # response_dict = dashscope.MultiModalConversation.call(model='qwen-vl-max',
    #                                                  messages=new_prompt)
    #call with local file
    response_dict = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                               messages=new_prompt, top_p=0.5, top_k=50)
    # response_dict = MultiModalConversation.call(model='qwen-vl-plus',
    #                                            messages=new_prompt)

    if response_dict.status_code == HTTPStatus.OK:
        response = response_dict[
            "output"]["choices"][0]["message"]["content"]
        print("CURI response:\n", response)

        # Plot the image with bounding boxes if the response contains actions.
        plot_image_with_bbox(local_file_path1, response)
    else:
        print(response_dict.code)  # The error code.
        print(response_dict.message)  # The error message.

    if return_response:
        return response

def multiple_multimodal_call(base_prompt, query, rounds = 10, log=True, return_response=False):

    """multiple rounds of  multimodal conversation call with CURIGPT.
    """
    base_prompt.append({"role": "user", "content": [{'image': local_file_path1}, {"text": query}]})

    for i in range(rounds):
        new_prompt = base_prompt
        # response = dashscope.MultiModalConversation.call(model='qwen-vl-max',
        #                                                  messages=new_prompt)
        #call with local file
        response_dict = MultiModalConversation.call(model=MultiModalConversation.Models.qwen_vl_chat_v1,
                                                   messages=new_prompt)
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

def get_curi_response_with_audio(api_key, base_url, user_input_filename, curigpt_output_filename,
                                 base_multimodal_prompt, rounds=10,
                                 prompt_append=False):

    """Get CURI response with audio input and output."""

    assistant = AudioAssistant(api_key, base_url, user_input_filename, curigpt_output_filename)

    inference_results = None
    if not prompt_append:
        # print("without prompt append")
        for i in range(rounds):
            assistant.record_audio()
            transcription = assistant.transcribe_audio()
            instruction = transcription
            response_dict = single_multimodal_call(base_multimodal_prompt, instruction, log=True, return_response=True)

            print("CURI audio response:\n", response_dict)
            # Parse the JSON string.
            try:
                output_data = json.loads(response_dict)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return

                # Go through each action and plot the bounding boxes.
            if output_data['robot_response'] is not None:
                response = output_data['robot_response']
                assistant.text_to_speech(response)

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



if __name__ == '__main__':
    # multiple_call_with_local_file()
    # rounds = int(input("How many rounds of conversation do you want? "))
    get_curi_response(base_multimodal_prompt, prompt_append=False)

    # api_key = "sk-59XTKMjGzgbgSJjpC9D770A52eBd4d68902223561eE3F242"
    # base_url = "https://www.jcapikey.com/v1"
    # user_input_filename = '/home/zhuoli/PycharmProjects/CuriGPT/assets/chat_audio/user_input.wav'
    # curigpt_output_filename = '/home/zhuoli/PycharmProjects/CuriGPT/assets/chat_audio/curigpt_output.mp3'
    #
    # get_curi_response_with_audio(api_key, base_url, user_input_filename, curigpt_output_filename,base_multimodal_prompt, rounds=10, prompt_append=False)