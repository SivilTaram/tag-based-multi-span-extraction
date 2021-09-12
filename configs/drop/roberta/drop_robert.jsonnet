{
    special_numbers:: [100, 1],
    multi_span_ignore_question:: false,
    multi_span_decoding_style:: "at_least_one",
    multi_span_training_style:: "soft_em",
    multi_span_prediction_method:: "viterbi",
    multi_span_labels:: {
        'O': 0,
        'I': 1
    },
	"pretrained_model": "roberta-large",
	"bert_dim": 1024,
    hidden_dim:: std.min($.bert_dim, 1024),
    default_ffn(input_dim, hidden_dim, output_dim, dropout = 0.1):: {
            "input_dim": input_dim,
            "num_layers": 2,
            "hidden_dims": [hidden_dim, output_dim],
            "activations": ["relu", "linear"],
            "dropout": [dropout, 0.0]
    },
    index_prediction_module(input_dim):: {
        "input_dim": input_dim,
        "num_layers": 1,
        "hidden_dims": 1,
        "activations": "linear"
    },
    heads:: {
        passage_span: {
            "type": "passage_span_head",
            "start_output_layer": $.index_prediction_module(input_dim = $.bert_dim),
            "end_output_layer": $.index_prediction_module(input_dim = $.bert_dim),
            "training_style": $.multi_span_training_style
        },
        question_span: {
            "type": "question_span_head",
            "start_output_layer": $.default_ffn(input_dim = 2 * $.bert_dim, hidden_dim = $.hidden_dim, output_dim = 1),
            "end_output_layer": $.default_ffn(input_dim = 2 * $.bert_dim, hidden_dim = $.hidden_dim, output_dim = 1),
            "training_style": $.multi_span_training_style
        },
        multi_span: {
            "type": "multi_span_head",
            "output_layer": $.default_ffn(input_dim = $.bert_dim, hidden_dim = $.hidden_dim, output_dim = std.length($.multi_span_labels)),
            "ignore_question": $.multi_span_ignore_question,
            "prediction_method": $.multi_span_prediction_method,
            "decoding_style": $.multi_span_decoding_style,
            "training_style": $.multi_span_training_style,
            "labels": $.multi_span_labels
        },
        arithmetic: {
            "type": "arithmetic_head",
            "output_layer": $.default_ffn(input_dim = 2 * $.bert_dim, hidden_dim = $.hidden_dim, output_dim = 3),
            "special_numbers": $.special_numbers,
            "special_embedding_dim": $.bert_dim,
            "training_style": $.multi_span_training_style,
        },
        count: {
            "type": "count_head",
            "max_count": 10,
            "output_layer": $.default_ffn(input_dim = $.bert_dim, hidden_dim = $.hidden_dim, output_dim = self.max_count + 1),
        }
    },
    answer_field_generators:: {
        tagged_answer: {
            "type": "tagged_answer_generator",
            "ignore_question": $.multi_span_ignore_question,
            "labels": $.multi_span_labels
        },
        arithmetic_answer: {
            "type": "arithmetic_answer_generator",
            "special_numbers": $.special_numbers
        },
        count_answer: {
            "type": "count_answer_generator"
        },
        passage_span_answer: {
            "type": "span_answer_generator",
            "text_type": "passage"
        },
        question_span_answer: {
            "type": "span_answer_generator",
            "text_type": "question"
        }
    },
    "dataset_reader": {
        "type": "tbmse_drop",
        "tokenizer": {
            "type": "huggingface_transformers",
            "pretrained_model": $.pretrained_model
        },
        "answer_field_generators": {
            "tagged_answer": $.answer_field_generators.tagged_answer,
            "arithmetic_answer": $.answer_field_generators.arithmetic_answer,
            "count_answer": $.answer_field_generators.count_answer,
            "passage_span_answer": $.answer_field_generators.passage_span_answer,
            "question_span_answer": $.answer_field_generators.question_span_answer
        },
        "answer_generator_names_per_type": {
            "multiple_span": ["tagged_answer"],
            "single_span": ["tagged_answer", "passage_span_answer", "question_span_answer"],
            "number": ["arithmetic_answer", "count_answer", "passage_span_answer", "question_span_answer", "tagged_answer"],
            "date": ["arithmetic_answer", "passage_span_answer", "question_span_answer", "tagged_answer"]
        },
        "old_reader_behavior": true,
        "is_training": true,
        "pickle": {
            "path": "../pickle/drop",
            "file_name": "all_heads_" + $.pretrained_model,
            "action": "load" # save / load / null
        }
    },
    "validation_dataset_reader": $.dataset_reader + {"is_training": false},
    "iterator": {
        "type": "basic",
        "batch_size": 2,
    },
    "model": {
        "type": "multi_head",
        "dataset_name": "drop",
        "pretrained_model": $.pretrained_model,
        local head_count = std.length(self.heads),
        "head_predictor": if head_count > 1 then $.default_ffn(input_dim = 2 * $.bert_dim, hidden_dim = $.bert_dim, output_dim = head_count) else null,
        local summary_module = {
            "input_dim": $.bert_dim,
            "num_layers": 1,
            "hidden_dims": 1,
            "activations": "linear"
        },
        "passage_summary_vector_module": summary_module,
        "question_summary_vector_module": summary_module,
        "heads": {
            "passage_span": $.heads.passage_span,
            "question_span": $.heads.question_span,
            "multi_span": $.heads.multi_span,
            "arithmetic": $.heads.arithmetic,
            "count": $.heads.count,
        }
    },
    "train_data_path": "drop_data/drop_dataset_train.json",
    "validation_data_path": "drop_data/drop_dataset_dev.json",
    "trainer": {
        "cuda_device": 0,
        "keep_serialized_model_every_num_seconds": 3600,
        "num_epochs": 35,
        "optimizer": {
            "type": "bert_adam",
            "lr": 1e-5,
        },
        "patience": 10,
        "summary_interval": 100,
        "validation_metric": "+f1",
        "num_steps_to_accumulate": 6,
    }
}
