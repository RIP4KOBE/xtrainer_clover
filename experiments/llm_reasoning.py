import dspy
from dspy.teleprompt import LabeledFewShot
import traceback
import traceback
import glob
import json
import os
import numpy as np


class GenerateRewardFromFeedBack(dspy.Signature):
    """Use the provided language instruction to write code for guiding a lower-level driving policy."""
    feedback = dspy.InputField()
    reward = dspy.OutputField()


class GenerateReward(dspy.Module):
    def __init__(self):
        super().__init__()

        self.predict = dspy.Predict(GenerateRewardFromFeedBack)

    def forward(self, feedback):
        prediction = self.predict(feedback=feedback)
        return dspy.Prediction(reward=prediction.reward)


def reformat_output_as_generator(reward):
    """
    Since `exec` cannot handle yield statements outside of function definitions,
    reformat the code to add a function definition.
    """
    lines = reward.split('\n')
    lines = [(' ' * 4) + line for line in lines]
    lines = ['def language_reward(self):'] + lines
    new_reward = '\n'.join(lines)
    return new_reward


class LLMEcotReasoning:
    def __init__(self, language_config, example_path):

        self.language_config = language_config
        self.reward = None
        self.example_path = example_path

    def load_examples(self):
        # Load instructions
        instructions_path = os.path.join(self.example_path, 'instructions.txt')
        with open(instructions_path, 'r') as f:
            instructions = f.read()
        instructions = instructions.split('\n')

        # Load demos
        demo_paths = glob.glob(f'{self.example_path}/demo*.py')
        demo_paths = sorted(demo_paths, key=lambda x: int(x.split('/')[-1][4:-3]))
        demos = []
        for demo_path in demo_paths:
            with open(demo_path, 'r') as f:
                demo = f.read()
            demos.append(demo)

        assert len(instructions) == len(demos), \
            f'Different number of demos and instructions, got {len(demos)} and {len(instructions)} respectively'

        # Build examples
        examples = [dspy.Example(feedback=instruction, reward=demo) for (instruction, demo) in
                    zip(instructions, demos)]
        return examples


    def build_llm_module(self):
        # Configure LM
        self.llm = dspy.LM(
            model=self.language_config['model'],
            temperature=self.language_config['temperature'],
            max_tokens=self.language_config['max_tokens'],
            api_key=self.language_config['api_key'],
            base_url=self.language_config['base_url']
        )
        dspy.settings.configure(lm=self.llm)

        # Load examples
        examples = self.load_examples()

        # Build and compile module
        llm_module = GenerateReward()
        tp = LabeledFewShot(k=len(examples))
        self.llm_module = tp.compile(llm_module, trainset=examples)

    def generate_reward(self):
        """
        Produce language-conditioned plan
        """
        # Invoke LLM here
        self.build_llm_module()
        language_feedback = self.language_config['feedback']

        output = self.llm_module(feedback=language_feedback)
        reward = output.reward
        print(f'LANGUAGE FEEDBACK: {language_feedback}')

        try:
            print('LLM GENERATED CODE:')
            print(reward)
            reward = reformat_output_as_generator(reward)
            exec(reward, globals())
            self.reward = language_reward(self)
        except Exception:
            print('*' * 40)
            print('Exception encountered while executing LLM code, trace below')
            print('*' * 40)
            print(traceback.format_exc())
            print('*' * 40)
            import pdb; pdb.set_trace()

        return self.reward


if __name__ == '__main__':
    # Example usage
    example_path = '../assets/ecot_prompt/template'
    language_config = {
        'model': 'gpt-4o',
        'temperature': 0.2,
        'max_tokens': 1000,
        'api_key': "sk-mSyK58YNPAdpDcyL5cFb693b61Ff4fD8A367539e68C5898c",
        'base_url': 'https://www.jcapikey.com/v1',
        'feedback': 'Stay in the current lane.'
    }

    llm_reasoning = LLMEcotReasoning(language_config=language_config, example_path=example_path)
    language_reward = llm_reasoning.generate_reward()