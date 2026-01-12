import argparse
import functools
import json
import multiprocessing as multip
import pathlib
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from llmchat.chater import BaseAssistant
from paperextractor.convert import str_to_data
from paperextractor.postprocess import (
    add_doi_and_id,
    expand_dict_columns,
    filter_csv,
)

def get_csv_form_json(json_name, save_dir=".", csv_name="auto"):
    with open(json_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    msg = data["messages"]
    mark = data["text_mark"]

    msg2 = []
    for i in mark:
      if mark ==4:
        msg2.append(msg[i]["content"])

    assert len(msg2) >= 8, f"Error: {json_name} has {len(msg2)} messages, but expected at less 8."

    data_df = _data_init_to_csv(msg2)

    if csv_name == "auto":
        csv_name = pathlib.Path(json_name).stem
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        csv_name = save_dir / (csv_name + ".csv")

    if data_df is not None:
        data_df.to_csv(csv_name, index=True,encoding="utf-8")
        print(f"Save to {csv_name}")

def _data_init_to_csv(data):
    # data0~data4 分别为 LLM 每轮输出的 JSON 字典
    data0, data1, data2, data3, data4 = data
    phase_keys = set()
    records = []
    # 字符串转为dict
    data4 = str_to_data(data4, del_sign="")
    for name,info in data4.items():
        phases = info.get("Phases",{})
        if isinstance(phases,dict):
            for ph_n,ph_info in phases.items():
                if isinstance(ph_info,dict):
                    for k in ph_info.keys():
                        phase_keys.add(f"{ph_n}.{k}")
                else:
                    phase_keys.add(f"{ph_n}")
    phase_keys = sorted(phase_keys)

    for name, info in data4.items():
        row = {}
        row["material_name"] = name
        # 先处理常规字段
        for key, value in info.items():
            if key == "Phases":
                continue 
            elif isinstance(value, dict):
                # composition, Phase_ratio等dict用json字符串保存
                row[key] = str(value)
            elif isinstance(value, list):
                row[key] = ";".join(map(str, value))
            else:
                row[key] = value      
    # 处理Phases展开
        phases = info.get("Phases", {})
        for pk in phase_keys:
            ph, *attr = pk.split(".")
            if attr:
                # 需要找具体Phase下的属性
                attr = attr[0]
                ph_val = phases.get(ph, {})
                val = ""
                if isinstance(ph_val, dict):
                    val = ph_val.get(attr, "")
                row[f"Phases.{ph}.{attr}"] = val
            else:
                # Phase直接是字符串
                row[f"Phases.{ph}"] = phases.get(ph, "")

        records.append(row)
    # 字段顺序
    # 先主字段，再Phases相关
    main_keys = ["material_name", "source text", "composition", "Type_main", "All_phases", "Structure", "Phase_ratio", "ratio_type", "process_chain", "Yield_Strength"]
    all_cols = [k for k in main_keys if k in records[0]] + [k for k in sorted(records[0].keys()) if k not in main_keys]
    # 保证Phases列顺序紧跟main_keys后
    all_cols += [f"Phases.{pk.replace('.', '.')}" for pk in phase_keys if f"Phases.{pk.replace('.', '.')}" not in all_cols]

    df = pd.DataFrame(records)
    # 按顺序输出
    df = df.reindex(columns=[col for col in main_keys if col in df.columns] + [col for col in df.columns if col not in main_keys])
    return df

def data_init_to_csv(self:BaseAssistant):

    msg = self.get_res_dict()

    data = list(msg.values())

    data_df = _data_init_to_csv(data)

    return data_df


class HEA_assistant(BaseAssistant):
      to_table = data_init_to_csv

def chat_chain_get_msg(
    manuscript=None,
    support_si_dir= None,
    out_dir='output',
    del_old=True,
    debug= False,
    model_json= None,
    verbose = False,
    warm_start= False,
    warm_prompt= None
):
    '''Send a chain of question to LLM, to let it extract structural data from a paper.
       All answers are saved as json file during the process and will be coverted to csv when the chain finished'''
    # manuscript, support type in  [pdf,docx,txt,doc]
    if debug:
        del_old = False
        if model_json is None:
            model_json = pathlib.Path(manuscript).parent.parent/"0_model_json/raw.json"

            if not model_json.exists():
                manuscript_stem = pathlib.Path(manuscript).stem
                model_json = pathlib.Path(out_dir)/manuscript_stem/"0_model_json/raw.json"

            if not model_json.exists():
                return "Error: model_json not exists, please check the path."

        ba = HEA_assistant.from_json(model_json)

    elif warm_start:
        del_old = False
        if model_json is None:
           model_json = pathlib.Path(manuscript).parent.parent/"0_model_json/raw.json"

           if not model_json.exists():
                  manuscript_stem = pathlib.Path(manuscript).stem
                  model_json = pathlib.Path(out_dir)/manuscript_stem/"0_model_json/raw.json"

           if not model_json.exists():
                  return "Error: model_json not exists, please check the path."

        ba = HEA_assistant.from_json(model_json)

        if warm_prompt is not None:
            answeradd = ba.Q(str(warm_prompt), response_format={"type": "json_object"})
        if verbose:
            print("Warm start answer:", answeradd)

    else:

        ba = HEA_assistant(
            system="你是一个严谨的高熵合金材料专家助手, 负责提取文献中的作者研究的材料信息。"
                   "总体目标是抽取每一个独立材料的组分、工艺、性质信息。注意永远以英文回答，"
                   "在所有提取结果中，将来自文章内部的引用符号（单、双引号）统一替换为反引号 `,"
                   "将来自文章内部的冒号符号统一替换为斜线 /  , 防止json解析错误。",
            offer_engine="DEEPSEEK",
            save_msg = {"root_dir":out_dir,}
        )

        ba.read_sci_total(manuscript,
                          sparse_engine="uni_sparser",
                          support_si_dir=support_si_dir,
                          left_suffix_name=[".pdf",".txt",".doc",".docx"],
                          remove_dir_name=["si_sparse_txt_pypdf2"],
                          verbose=True, 
                          save_to_dir=True,
                          tail_ask=True,
        )


        answer0 = ba.Q(
        """阅读文献, 判断正文中作者是否研究了高熵合金材料的性能, 并提供了所研究材料的结构和性质信息。
        回答格式如下, 不要返回其他解释说明信息：

         - 如果是, 返回 "Yes, the author studied the phase structure and properties of High Entropy alloys"
         - 如果不是, 返回 "No, the author did not study the phase structure and properties of High Entropy alloys"

        """,
        )

        if verbose:
            print("Answer0:", answer0)


        if "Yes" not in answer0 and "yes" not in answer0:
            return answer0



        answer1 = ba.Q(
        """
        请按以下规则提取文章中作者研究到的主要高熵合金材料名称信息：
        - 仅提取文章研究的高熵合金材料。引用的其他文章中的材料、工艺中用到的其他试剂等均无需提取。
        - 材料名称中的所有: 替换为/ 。
        - 若某材料有多个别名或缩写, 只返回一个能够与其他材料区分, 信息最丰富的的名称, 如AlCoCrFeNi2.1。名义成分和实际成分使用名义成分。
        - 若材料名称中的部分化学式、元素名或成分的数值比例为可变量, 遍历文中所有可替换值, 在原有名称后添加标记, 拆分获得多个材料名称。具体如下
          1.如AlCoCrFeNix, x=2.1, 1.5, 则提取 AlCoCrFeNix(x=2.1),AlCoCrFeNix(x=1.5) 两种材料, 而不是 Li1-mC2O3(m=0.1,0.2,0.3),即每个名称的变量被明确写出需要代入的单一值。
          2.如Li2XO3, X=F,Cl,I, 则提取 Li2XO3(X=F), Li2XO3(X=Cl), Li2XO3(X=I) 三种材料, 而不是直接使用 Li2XO3(X=F,Cl,I)。
        - 对于提取的合金材料，按元素的摩尔比拆分成分
        - 提取类型信息(Type_main)，限制在三类中:SS(Solid-solution,固溶体),IM(Intermetallic compounds,金属间化合物),SS+IM(前两者的结合，即材料中即包含固溶体相，也包含金属间化合物相)
        - 提取主要相组成,例如FCC,BCC,HCP,B2,laves...

        严格输出格式：

        ```json
        {
            "AlCoCrFeNix(x=2.1)": {
            "source text": "xxxxxxxxxxxx",
            "composition": {"Al": "1", "Co": "1", "Cr": "1","Fe":"1", "Ni": "2.1"},
            "Type_main":"SS+IM",
            "Phase":{"Phase1":"FCC","Phase2":"B2"}
            },
            "AlCoCrFeNix(x=1.5)": {
            "source text": "xxxxxxxxxxxx",
            "composition": {"Al": "1", "Co": "1", "Cr": "1","Fe":"1", "Ni": "1.5"},
            "Type_main":"SS+IM",
            "Phase":{"Phase1":"FCC","Phase2":"B2"}
            }
        }
        ```

        """,response_format={"type": "json_object"})


        #if verbose:
            #print("Answer1:", answer1)

        answer2 = ba.Q(
        """
        根据上述提取的信息,
        首先如果含有包含可变比例的括号内容,将变量的值代入原成分式，例如"AlCoCrFeNix(x=2.1)"变为"AlCoCrFeNi2.1"
        
        接下来，如果材料具有多个相,即phase内不只有一组键对:
        - 根据原文补充各相形成的组织类型,例如层状(lammela/layer structure),网络状(reticulate structure),弥散析出(diffuse precipitation structure)
        - 提取各相之间的比例信息，以及比例类型，例("1:1"),("Volume Fraction","Weight Fraction")
        并添加一个信息，总体描述合金的相组成，"All_phases",值为"Phases"中所有的values通过+连接组成的字符串,例如如果"Phases":{"Phase1":"FCC","Phase2":"B2"},则"All_phases":"FCC+B2"

        然后提取上述材料每一个相内部详细特征。

        字段如下：
          - 富集元素(elements):例如("Co,Cr,Fe"),用一个字符串表述即可,用逗号分割元素
          - 晶粒尺寸(grain_size):"6.5±1.3 μm",统一换算成微米μm尺度
          - 性能特征(property):提取对该相的描述关键词，例如软硬、高延展、高强度等,用一个字符串表述即可
          - 微结构(inner_structure):相的内部微结构，例如纳米析出相,同时提取其类型、比例、尺寸等("nanoprecipitates":{"type":"L12","Fraction":"20%")

         严格输出格式：
        ```json
        {
            "AlCoCrFeNi2.1": {
            "source text": "xxxxxxxxxxxx",
            "composition": {"Al": "1", "Co": "1", "Cr": "1","Fe":"1", "Ni": "2.1"},
            "Type_main":"SS+IM",
            "Phases":{
                      "Phase1":{"phase_main":"FCC",
                                "elements":"Co,Cr,Fe",
                                "grain_size":"0.73 ± 0.12 μm",
                                "properties":"hard",
                                "inner_structure":{"nanoprecipitates":{"type":"L12","volume_fraction":"25%"}}
                                }
                      "Phase2":{"phase_main":"B2",
                                "elements":"Al,Ni",
                                "grain_size":"0.66 ± 0.15 μm",
                                "properties":"soft",
                                "inner_structure":{"nanoprecipitates":{"type":"BCC","volume_fraction":"19.32%"}}
                                }
                    },
            "All_phases":"FCC+B2",
            "Structure":"layer_structure",
            "Phase_ratio":{"Phase1":"1","Phase2":"1"}
            "ratio_type":"volume_fraction"
            },
        }
        ```
        """,response_format={"type": "json_object"})


        #if verbose:
        #    print("Answer2:", answer2)


        answer3 = ba.Q(
        """
        工艺流程是指制备、加工材料的工艺动作的集合。一种工艺流程包含若干工艺动作。一种工艺流程内部，可能会调整具体工艺动作的不同的参数或改变、增减部分动作，实现性能优化。但这种个别动作、参数的改变，统称为一类工艺流程。
        两种工艺流程之间，工艺动作甚至原理可能完全不同, 如液相法(liquid-phase technique)与机械法(mechanical milling technique)等。
        按照工艺施加顺序, 提取文中的所有制备、加工工艺名称及细节。

        要求如下：
        - 工艺命名规范为：<工艺名称>_<主参数> , 例如"sintering_800C_12h"。附加的1~2个参数用以区分文中同名但不同细节的工艺。
        - 根据工艺路线,为经过工艺处理的材料添加版本号,每一个版本变为一个格式相同的字典：
          1、如果该命名的材料在工艺流程中首次出现, 命名版本为v1。
          2. 如果该材料经过了新的工艺步骤,且化合成分没有变化, 创建子版本,如v1->v2。例如: 首次制备出现时, 命名为AlCoCrFeNi2.1_v1, 经过重结晶工艺, 命名为AlCoCrFeNi2.1_v2。
          3. 对每一个版本,均补充前面问答已经获取的所有信息,如果某一个信息经过工艺未发生变化则沿用上一版本的值，否则则更新为新的值
          4. All_phases的值不随工艺发生变化
          5.每一个版本加上经历的工艺流程,"process_chain":"[...]",值为经历过的所有工艺名称组成的列表
        - 为了书写简便, 提取结果的摄氏度单位统一设置为"C"。

        输出格式：
        ```json
        {

            "AlCoCrFeNi2.1_v1": {
            "source text": "xxxxxxxxxxxx",
            "composition": {"Al": "1", "Co": "1", "Cr": "1","Fe":"1", "Ni": "2.1"},
            "Type_main":"SS+IM",
            "Phases":{.... },
            "All_phases":"FCC+B2",
            "Structure":"layer_structure",
            "Phase_ratio":{"Phase1":"1","Phase2":"1"}
            "ratio_type":"volume_fraction"
            "process_chain":[]
            },
            "AlCoCrFeNi2.1_v2": {
            "source text": "xxxxxxxxxxxx",
            "composition": {"Al": "1", "Co": "1", "Cr": "1","Fe":"1", "Ni": "2.1"},
            "Type_main":"SS+IM",
            "Phases":{.... },
            "All_phases":"FCC+B2",
            "Structure":"layer_structure",
            "Phase_ratio":{"Phase1":"1","Phase2":"1"}
            "ratio_type":"volume_fraction"
            "process_chain":["Recrystalize_1000C_1h"]
            },
             "AlCoCrFeNi2.1_v3": {
            "source text": "xxxxxxxxxxxx",
            "composition": {"Al": "1", "Co": "1", "Cr": "1","Fe":"1", "Ni": "2.1"},
            "Type_main":"SS+IM",
            "Phases":{.... },
            "All_phases":"FCC+B2",
            "Structure":"layer_structure",
            "Phase_ratio":{"Phase1":"1","Phase2":"0.8"}
            "ratio_type":"volume_fraction"
            "process_chain":["Recrystalize_1000C_1h","Aged_600C_10h"]
            }
        }
        ```

        """,response_format={"type": "json_object"})

        #if verbose:
        #   print("Answer3:", answer3)

        answer4 = ba.Q(
        """
        提取每个材料版本的力学性能, 要求格式如下：

        主字段性质: Hardness、Yield Strength, Tensile Strength,Fracture Toughness
        每一个性能均表示成"性能名称":"数值 (单位)"，例如"Yield_Strength":"270 (Mpa)"

        输出格式示例如下：
        ```json
        {
           "AlCoCrFeNi2.1_v1": {
            "source text": "xxxxxxxxxxxx",
            "composition": {"Al": "1", "Co": "1", "Cr": "1","Fe":"1", "Ni": "2.1"},
            "Type_main":"SS+IM",
            "Phases":{.... },
            "All_phases":"FCC+B2",
            "Structure":"layer_structure",
            "Phase_ratio":{"Phase1":"1","Phase2":"1"}
            "ratio_type":"volume_fraction"
            "process_chain":[]
            "Yield_Strength":"270 (Mpa)"
            },
             "AlCoCrFeNi2.1_v2":{...},
            ...
        }

        ```
        """
        ,response_format={"type": "json_object"})

        if verbose:
            print("Answer4:", answer4)

    # 输出json
    ba.to_loop(del_old=del_old,loop=0)
    # 输出为表格
    csv_name = ba.csv_to_loop(loop=0)
    csv_name = pathlib.Path(csv_name)
    print(f"Final CSV saved to: {csv_name}")
    return (f"Final CSV saved to: {csv_name}")

def chat_chain_get_msg_except(manuscript, **kwargs):#错误处理，**kwargs动态参数，接受原调用的所有参数
    try:
        return chat_chain_get_msg(
            manuscript=manuscript,
            **kwargs)
    except Exception as e:
        raise e
        print(f"An error occurred while processing {manuscript}: {e}")
        return None

def chat_chain_get_msg_pool(manuscript:list, support_si_dir, out_dir, del_old=True,
                            verbose=False, n_jobs=8,debug=False):
    #多进程批量处理文献列表
    func = functools.partial(chat_chain_get_msg_except,
        support_si_dir=support_si_dir,
        out_dir=out_dir,
        del_old=del_old,
        verbose=verbose,debug=debug
    )#固定部分参数生成新函数，从而只传入manuscript即可调用

    results = []
    finished = []
    #存储处理结果以及已完成的文件名
    try:
        with multip.Pool(n_jobs) as pool:
            async_results = []
            for i in tqdm(manuscript, desc="Processing manuscripts", unit="file", disable=not verbose):
                async_results.append((i, pool.apply_async(func, (i,))))

            for n,(i, result) in enumerate(async_results):
                print(f"\nProcessing manuscript {n+1}/{len(manuscript)}: {i}")
                res = result.get()
                results.append(f"{i}, {res}")
                finished.append(i)

        print("All manuscripts processed successfully.")

    except Exception as e:
        raise e
        print(f"An error occurred during processing: {e}")
    finally:
        pool.close()
        pool.join()
        #关闭进程池等待子进程结束
        remaining = [i for i in manuscript if i not in finished]


        # 保存已经运行完的词条
        with open("llm_finished_items.txt", "w", encoding="utf-8") as f:
            for item in finished:
                f.write(str(item) + "\n")
        # 保存剩余的词条
        with open("llm_remaining_items.txt", "w", encoding="utf-8") as f:
            for item in remaining:
                f.write(str(item) + "\n")

        with open("llm_results.txt", "w", encoding="utf-8") as f:
            for item in results:
                f.write(str(item) + "\n")

        print(f"Finished processing {len(finished)} files, {len(remaining)} files remaining.")
        print(f"Results saved to llm_results.txt")
        print("finished items to llm_finished_items.txt")
        print("remaining items to llm_remaining_items.txt")

    return results




def main():
    import argparse
    #设置解析器以及描述
    parser = argparse.ArgumentParser(description="Extract solid electrolyte data from manuscript PDF.")
    parser = add_args(parser)#添加自定义参数
    args = parser.parse_args()#解析命令行参数
    run(args)

def add_args(parser):
    #集中定义所有命令行参数，action：该参数存在时存储true/false
    parser.add_argument("-m","--manuscript", type=str, nargs="*",default=None, help="Path to the manuscript PDF/docx/txt file.")
    parser.add_argument("-pf","--paths_file", type=str,  help="file containing paths to manuscript files, one per line.",default=None)
    parser.add_argument("-sid","--support_si_dir", type=str, default=None, help="Directory containing supporting information files.")
    parser.add_argument("-o","--out_dir", type=str, default=".", help="Root directory for saving outputs.")
    parser.add_argument("-del","--del_old", action="store_false", help="Delete old intermediate files.")
    parser.add_argument("-v","--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed output.")
    parser.add_argument("-n","--n_jobs", type=int, default=8, help="Number of parallel jobs to run (default: 8).")
    parser.add_argument("-c","--config", type=str, default=None, help="Path to configuration json file (optional).")
    
    return parser

def run(args):
    #处理参数并调用文献处理相关函数
    if args.config:
        # Load configuration from JSON file if provided
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)#读取json配置
            args.__dict__.update(config)  # Update args with config values
        except Exception as e:
            print(f"Error loading config file: {e}")

    for k,v in args.__dict__.items():
        print(f"Argument {k} = {v}")#打印所有参数值

    if args.paths_file:
        with open(args.paths_file, 'r') as f:
            args.manuscript = [line.strip() for line in f if line.strip()]#从文本文件读取路径列表，并过滤空行、去除空格

    if not args.manuscript:
        raise ValueError("No manuscript provided. Please specify a manuscript file or a path file containing manuscript paths.")

    if isinstance(args.manuscript, str):
        args.manuscript = [args.manuscript]#manuscript转换为列表，只输入单个文件

    result = chat_chain_get_msg_pool(
        manuscript=args.manuscript,
        support_si_dir=args.support_si_dir,
        out_dir=args.out_dir,
        del_old=args.del_old,
        verbose=args.verbose,
        n_jobs=args.n_jobs,
        debug=args.debug
    )#批量提取数据


if __name__ == "__main__":

    main()


# if __name__ == "__main__":
#     # Argument support_si_dir = /dp-library/sulfide-solid-electrolyte/support01
#     # Argument out_dir = /root/paperextractor/soild_data_root2
#     # Argument del_old = False
#     # Argument verbose = False
#     # Argument debug = True
#     # Argument n_jobs = 1

#     chat_chain_get_msg_pool(
#         manuscript=["/root/paperextractor/soild_data_root3/10_1002-aenm_201501590/manuscript_pdf/10_1002-aenm_201501590.pdf"],
#         support_si_dir="/dp-library/sulfide-solid-electrolyte/support02",
#         out_dir="/root/paperextractor/soild_data_root3",
#         del_old=False,
#         verbose=False,
#         n_jobs=1,
#         debug=True
#     )








