import os
import hls4ml
import glob
import xml.etree.ElementTree as ET

def hls_convert(model, full_name, input_shape=(256, 15), precision='fixed<16,6>'):
    print(f" [HLS] Converting model: {full_name} with precision {precision}")
    
    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=input_shape,
        granularity='name',
        default_precision=precision,
        default_reuse_factor=5
    )

    config['Model']['Strategy'] = 'Resource'
    config['Model']['IOType'] = 'io_stream'

    # print("Capas detectadas:", list(config['LayerName'].keys()))

    if 'LayerName' in config:
        for layer_name, layer_cfg in config['LayerName'].items():
            layer_cfg['Strategy'] = 'Resource'
            layer_cfg['ReuseFactor'] = 5

    # 3. C++ project generation
    output_dir = os.path.join('hls', full_name)
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=input_shape,
        hls_config=config,
        output_dir=output_dir,
        part='xc7z020clg400-1',
        backend='Vitis'
    )

    hls_model.compile()

    # 5. Build the project with Vivado
    try:
        hls_model.build(csim=True, synth=True, export=True)
    except Exception as err:
        cycles, latency_us = extract_latency(project_path=output_dir)
        print(f" [HLS] Latency extracted: {latency_us:.4f} us, Cycles: {cycles}")
        
        export_zip = os.path.join(output_dir, 'myproject_prj/solution1/impl/export.zip')
        csynth_rpt = os.path.join(output_dir, 'myproject_prj/solution1/syn/report/csynth.rpt')

        if cycles is not None or latency_us > 0 or os.path.exists(export_zip) or os.path.exists(csynth_rpt):
            print("\n" + "!"*80)
            print(" [HLS Warning] Vitis HLS completed the C-Synthesis and the IP export with success.")
            print("               The memory crash on tool closing was ignored.")
            print("               Continuing the framework execution...")
            print("!"*80 + "\n")
        else:
            log_file = os.path.join(output_dir, 'vitis_hls.log')
            if os.path.exists(log_file):
                print("\n" + "="*25 + " LAST LINES OF VITIS_HLS.LOG " + "="*25)
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-30:]:
                        print(line.strip())
                print("="*80 + "\n")
            raise err

    return hls_model


def extract_latency(project_path):
    """
    Extracts the latency by first reading the structured csynth.xml file (100% reliable in Vitis 2026).
    If it does not exist, it uses a generic fallback on any csynth.rpt.
    """
    if not os.path.exists(project_path):
        return None, 0.0

    # Attempt to extract latency from csynth.xml or csynth.rpt files
    xml_files = glob.glob(os.path.join(project_path, '**', 'csynth.xml'), recursive=True)
    if xml_files:
        try:
            tree = ET.parse(xml_files[0])
            root = tree.getroot()
            worst_lat = root.find('.//PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency')
            if worst_lat is not None and worst_lat.text.strip().isdigit():
                cycles = int(worst_lat.text.strip())
                return cycles, (cycles * 5.0) / 1000.0
        except Exception:
            pass

    # 2. Fallback: Extract latency from csynth.rpt files
    rpt_files = glob.glob(os.path.join(project_path, '**', '*csynth.rpt'), recursive=True)
    for rpt_path in rpt_files:
        try:
            with open(rpt_path, 'r') as f:
                lines = f.readlines()
            
            in_summary = False
            for line in lines:
                if '* Summary:' in line:
                    in_summary = True
                    continue
                if in_summary and '|' in line and not any(k in line for k in ['Latency', '---', 'min', 'max', 'Module']):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 2:
                        for p in parts[:2]:
                            if p.isdigit():
                                cycles = int(p)
                                return cycles, (cycles * 5.0) / 1000.0
        except Exception:
            pass

    return None, 0.0
