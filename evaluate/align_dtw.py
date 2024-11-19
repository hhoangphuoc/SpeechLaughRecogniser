import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
import torch
import jiwer
import whisper_timestamped as whisper_ts
from utils.transcript_process import clean_transcript_sentence, transform_number_words
from tqdm import tqdm
import matplotlib.pyplot as plt

#===================================================================
# REMOVE TF WARNINGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)
#===================================================================

def align_dtw(
        audio_path, 
        ref_transcript,
        model_name="openai/whisper-small",
        cache_dir="../ref_models/pre_trained", #If used fine-tuned model: ../vocalwhisper/speechlaughwhisper-subset-10
        plot_alignment_dir="../alignment_transcripts/plots",
        alignment_plot_name="laughing_word_miss",
        alignment_json_name="laughing_word_miss.json"
):  
    if not os.path.exists(plot_alignment_dir):
        os.makedirs(plot_alignment_dir)
    
    # check GPU availability    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plot_alignment_path = os.path.join(plot_alignment_dir, alignment_plot_name)
    alignment_json_path = os.path.join(plot_alignment_dir, alignment_json_name)

    model = whisper_ts.load_model(
        name=model_name,
        backend="transformers",
        # download_root=cache_dir
    )

    result = whisper_ts.transcribe(
        model=model,
        audio=audio_path,
        task="transcribe",
        language="en",
        plot_word_alignment=plot_alignment_path,
        remove_punctuation_from_words=True,
        detect_disfluencies=True #to indicate position having laughter, speechlaugh
        
    )
    
    # ref_transcript = clean_transcript_sentence(ref_transcript) # REMOVE EMPTY STRINGS, MULTIPLE SPACES, ALREADY LOWERCASE
    ref_transcript = jiwer.RemoveMultipleSpaces()(ref_transcript)

    # HYP: Transform number words to their numerical values
    hyp_transcript = transform_number_words(result["text"], reverse=True)
    hyp_transcript = clean_transcript_sentence(hyp_transcript)
    hyp_transcript = jiwer.RemovePunctuation()(hyp_transcript) # ALSO REMOVE PUNCTUATION

    # jiwer_alignment = jiwer.process_words(ref_transcript, hyp_transcript)
    
    # save result to json
    output_dict = {}
    output_dict['ref'] = ref_transcript
    output_dict['hyp'] = hyp_transcript
    # output_dict['ref_alignment'] = jiwer_alignment.references
    # output_dict['hyp_alignment'] = jiwer_alignment.hypotheses
    output_dict['segments'] = result["segments"]
    with open(alignment_json_path, "w") as f:
        json.dump(output_dict, f)

    print(f"Alignment plot saved to: {plot_alignment_path} \nand result to: {alignment_json_path}")


if __name__ == "__main__":

    examples = {
        "laughing_word_miss": {
            "original_audio": "sw02325A_25763475_270175625.wav",
            "audio_path": "../examples/alignments/laughing_word_miss.wav",
            "ref_transcript": "uh you know they were FILTHY when i would get home now i understand kids go out and play and they get DIRTY but i mean filthy i am talking sand in the EARS and the EYES and the HAIR and the and i was like gosh and then"
        },
        "laughing_word_hit": {
            "original_audio": "sw02297A_533487375_54041475.wav",
            "audio_path": "../examples/alignments/laughing_word_hit.wav",
            "ref_transcript": "only because i am i am too cheap to pay somebody uh twice the value of the oil and the FILTER DO IT"
        },
        "laughing_word_hit_miss": {
            "original_audio": "sw02241A_4284225_433435875.wav",
            "audio_path": "../examples/alignments/laughing_word_miss(getting)_hit(this).wav",
            "ref_transcript": "and they said well we just plan on spending the rest of our lives just GETTING THIS property developed"
        },
        "laughing_word_backchannel_hit": {
            "original_audio": "sw02548B_5221_6610875.wav",
            "audio_path": "../examples/alignments/laughing_word_backchannel_hit.wav",
            "ref_transcript": "YEAH"
        },
        "laughter_intext_miss": {
            "original_audio": "sw04357B_25400125_2594545.wav",
            "audio_path": "../examples/alignments/laughter_intext_miss.wav",
            "ref_transcript": "yeah and uh most people do not want to live that way today [LAUGHTER] in this country [LAUGHTER]"
        },
        "laughter_hit": {
            "original_audio": "sw04057B_131815_1424525.wav",
            "audio_path": "../examples/alignments/only_laughter_hit.wav",
            "ref_transcript": "[LAUGHTER]"
        },
        "laughter_substitution_hit": {
            "original_audio": "sw04323A_1670465_170344625.wav",
            "audio_path": "../examples/alignments/laughter_substitution_hit.wav",
            "ref_transcript": "[LAUGHTER]" #HYP: hehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehehe
        }
    }
    for example_name, example in tqdm(examples.items(), desc="Processing aligning examples"):
        align_dtw(
            model_name="openai/whisper-small",
            plot_alignment_dir="../alignment_transcripts/plot_alignments",
            audio_path=example["audio_path"],
            ref_transcript=example["ref_transcript"],
            alignment_plot_name=example_name,
            alignment_json_name=f"{example_name}.json"
        )
