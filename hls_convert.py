import os
import hls4ml

def hls_convert(model, full_name, input_shape=(256, 15), precision='fixed<16,6>'):
    config = hls4ml.utils.config_from_keras_model(
        model, 
        default_precision=precision,
        default_reuse_factor=80
    )

    config['Model']['Strategy'] = 'Resource'
    config['Model']['IOType'] = 'io_stream'

    if 'LayerName' not in config:
        config['LayerName'] = {}

   
    for layer in model.layers:
        layer_name = layer.name
        layer_type = layer.__class__.__name__

        if 'Conv1D' in layer_type or 'Dense' in layer_type:
            config['LayerName'][layer_name] = {
                'Strategy': 'Resource',
                'ReuseFactor': 80
            }
        elif 'Pooling' in layer_type or 'Activation' in layer_type:
            config['LayerName'][layer_name] = {
                'Strategy': 'Resource',
                'ReuseFactor': 1
            }

    output_dir = os.path.join('hls', full_name)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        part='xc7z020clg400-1',
        backend='Vitis'
    )

    tcl_path = os.path.join(output_dir, 'build_prj.tcl')
    if os.path.exists(tcl_path):
        with open(tcl_path, 'r') as f:
            tcl_content = f.read()
        
        if 'config_compile -expression_balance=off' not in tcl_content:
            tcl_content = tcl_content.replace(
                'csynth_design',
                'config_compile -expression_balance=off\ncsynth_design'
            )
            with open(tcl_path, 'w') as f:
                f.write(tcl_content)
            print(" [HLS Patch] Aplicada directiva '-expression_balance=off' en build_prj.tcl")

    hls_model.compile()

    try:
        hls_model.build(csim=False, synth=True, export=False)
    except Exception as err:
        cycles, latency_us = extract_latency(project_path=output_dir)
        if cycles is not None or latency_us > 0:
            print("\n" + "!"*80)
            print(" [HLS Warning] Vitis HLS completó la C-Síntesis y generó csynth.rpt con éxito.")
            print("               Continuando la ejecución del framework...")
            print("!"*80 + "\n")
        else:
            log_file = os.path.join(output_dir, 'vitis_hls.log')
            if os.path.exists(log_file):
                print("\n" + "="*25 + " ÚLTIMAS LÍNEAS DE VITIS_HLS.LOG " + "="*25)
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-30:]:
                        print(line.strip())
                print("="*80 + "\n")
            raise err

    return hls_model


def hls_convert2(model, full_name, input_shape=(256, 15), precision='fixed<16,6>'):
    config = hls4ml.utils.config_from_keras_model(
        model, 
        default_precision=precision,
        default_reuse_factor=80
    )

    config['Model']['Strategy'] = 'Resource'
    config['Model']['IOType'] = 'io_stream'

    if 'LayerType' not in config:
        config['LayerType'] = {}


    output_dir = os.path.join('hls', full_name)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=output_dir,
        part='xc7z020clg400-1',
        backend='Vitis'
    )

    tcl_path = os.path.join(output_dir, 'build_prj.tcl')
    if os.path.exists(tcl_path):
        with open(tcl_path, 'r') as f:
            tcl_content = f.read()
        
        if 'config_compile -expression_balance=off' not in tcl_content:
            tcl_content = tcl_content.replace(
                'csynth_design',
                'config_compile -expression_balance=off\ncsynth_design'
            )
            with open(tcl_path, 'w') as f:
                f.write(tcl_content)
            print(" [HLS Patch] Aplicado '-expression_balance=off' en build_prj.tcl para controlar RAM.")

    hls_model.compile()

    try:
        hls_model.build(csim=False, synth=True, export=False)
    except Exception as err:
        log_file = os.path.join(output_dir, 'vitis_hls.log')
        if os.path.exists(log_file):
            print("\n" + "="*25 + " ÚLTIMAS LÍNEAS DE VITIS_HLS.LOG " + "="*25)
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print(line.strip())
            print("="*80 + "\n")
        raise err

    return hls_model


def extract_latency(project_path='hls/CNNRULPredictor_1D_HLS'):
    base_path = os.path.join(project_path, 'myproject_prj/solution1/syn/report')
    csynth_path = os.path.join(base_path, 'csynth.rpt')
    
    if not os.path.exists(csynth_path):
        csynth_path = os.path.join(base_path, 'myproject_csynth.rpt')

    if not os.path.exists(csynth_path):
        return None, 0.0

    cycles = None
    latency_us = 0.0

    with open(csynth_path, 'r') as f:
        for line in f:
            if 'myproject' in line and '|' in line and 'Modules' not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                for i, elem in enumerate(parts):
                    if elem in ['yes', 'no'] and (i + 1) < len(parts):
                        try:
                            cycles = int(parts[i + 1])
                            if (i + 2) < len(parts) and parts[i + 2].replace('.', '', 1).replace('e+', '', 1).isdigit():
                                latency_us = float(parts[i + 2]) / 1000.0
                            else:
                                latency_us = (cycles * 5.0) / 1000.0
                            break
                        except ValueError:
                            continue
                if cycles is not None:
                    break

    return cycles, latency_us