from traitlets import default
from .bert import BertForSequenceClassification
from .bert.modeling_bert import BertGetLabelWord, BertUseLabelWord, BertDecouple, BertForMaskedLM
from .gpt2 import GPT2DoubleHeadsModel
from .gpt2.modeling_gpt2 import GPT2UseLabelWord

from .roberta import RobertaForSequenceClassification 
from .roberta.modeling_roberta import RobertaUseLabelWord

#from transformers import RobertaForMaskedLM, BartForConditionalGeneration, T5ForConditionalGeneration

from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
#from .roberta.modeling_roberta_plus import RobertaForMaskedLM,RobertaModel
from transformers import RobertaForMaskedLM,RobertaModel
class RobertaForPrompt(RobertaForMaskedLM):
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        parser.add_argument("--init_answer_words", type=int, default=1, )
        parser.add_argument("--init_type_words", type=int, default=1, )
        parser.add_argument("--init_answer_words_by_one_token", type=int, default=0, )
        parser.add_argument("--use_template_words", type=int, default=1, )
        return parser

class RobertaForPtuningv2(RobertaModel):
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        parser.add_argument("--init_answer_words", type=int, default=1, )
        parser.add_argument("--init_type_words", type=int, default=1, )
        parser.add_argument("--init_answer_words_by_one_token", type=int, default=0, )
        parser.add_argument("--use_template_words", type=int, default=1, )
        return parser

class BartRE(BartForConditionalGeneration):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        return parser
        
class T5RE(T5ForConditionalGeneration):
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--use_prompt", type=bool, default=True, help="Whether to use prompt in the dataset.")
        return parser




