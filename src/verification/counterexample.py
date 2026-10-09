def extract(obligation, model, context, input_encoding):
    index = [model.eval(i, model_completion=True).as_long() for i in obligation.indices]
    declared = {fn.get_id() for fn in model.decls()}
    functions = {
        ref.label(): str(model[fn])
        for ref, fn in context.functions.items()
        if fn.get_id() in declared
    }
    samples = {}
    for ref, tensor in input_encoding.tensors.items():
        zero = (0,) * len(tensor.shape)
        samples[ref.label()] = {
            "index": zero,
            "value": str(model.eval(tensor.at(zero), model_completion=True)),
        }
    return {
        "obligation": obligation.name,
        "local_output_index": index,
        "single_value": str(model.eval(obligation.single_value, model_completion=True)),
        "distributed_value": str(
            model.eval(obligation.distributed_value, model_completion=True)
        ),
        "free_function_interpretations": functions,
        "original_input_samples": samples,
        "model_smt2": model.sexpr(),
        "reconstruction": "Bound local inputs follow the recorded input substitutions for every valid index.",
    }
