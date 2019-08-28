from op_utils.sade_utils import generate_tools,generate_process,generate_reactionset
from get_flags import Flags
fl=Flags()
ngene=fl.dim
ratio=fl.ratio
ind=fl.it_value
np=fl.num_pro
cesspara=fl.pro_parameter
reaction=fl.reaction
reacline=fl.reacline

generate_reactionset(ngene,reaction,reacline)

#
generate_tools(num_gene=ngene,value=ind,low=ratio[0],high=ratio[1])
generate_process(numpro=np,parameter=cesspara)
#

with open('op_utils/script.py','r',encoding='utf-8')as f:
    lines=f.readlines()
    f1=open('main.py','w',encoding='utf-8')
    for line in lines:
        f1.write(line)
    f1.close()

with open('main.py','r',encoding='utf-8') as f:# 设定SADE的setting

    lines=[]
    for line in f:
        lines.append(line)
    f.close()
    tool=open('prepare_setting/setting.txt','r',encoding='utf-8')
    wlines=tool.readlines()
    length1=len(wlines)         #sade setting 用掉的行数
    for ind,line in enumerate(wlines):
        lines.insert(16+ind,"%s"%line)
    s=''.join(lines)
    f1=open('main.py','w',encoding='utf-8')
    f1.write(s)
    f1.close()

with open('main.py','r',encoding='utf-8') as f:# 设定第初始化反应的setting
    lines=[]
    for line in f:
        lines.append(line)
    f.close()
    tool=open('prepare_setting/reaction_setting.txt','r',encoding='utf-8')
    wlines=tool.readlines()
    length2=len(wlines)         #sade setting 用掉的行数
    for ind,line in enumerate(wlines):
        lines.insert(80+ind+length1,"        %s"%line)
    s=''.join(lines)
    f1=open('main.py','w',encoding='utf-8')
    f1.write(s)
    f1.close()

with open('main.py','r',encoding='utf-8') as f:#设定第一次多进程的setting
    lines=[]
    for line in f:
        lines.append(line)
    f.close()
    tool=open('prepare_setting/process_setting.txt','r',encoding='utf-8')
    wlines=tool.readlines()
    length3=len(wlines)
    for ind,line in enumerate(wlines):
        lines.insert(90+ind+length1+length2,"%s"%line)     #注意设定进程的空格以及行数
    s=''.join(lines)
    f1=open('main.py','w',encoding='utf-8')
    f1.write(s)
    f1.close()

with open('main.py','r',encoding='utf-8') as f:# 设定第初始化反应的setting
    lines=[]
    for line in f:
        lines.append(line)
    f.close()
    tool=open('prepare_setting/reaction_setting.txt','r',encoding='utf-8')
    wlines=tool.readlines()
    length4=len(wlines)         #sade setting 用掉的行数
    for ind,line in enumerate(wlines):
        lines.insert(175+ind+length1+length2+length3,"            %s"%line)
    s=''.join(lines)
    f1=open('main.py','w',encoding='utf-8')
    f1.write(s)
    f1.close()

with open('main.py','r',encoding='utf-8') as f:#设定第一次多进程的setting
    lines=[]
    for line in f:
        lines.append(line)
    f.close()
    tool=open('prepare_setting/process_setting.txt','r',encoding='utf-8')
    wlines=tool.readlines()
    length5=len(wlines)
    for ind,line in enumerate(wlines):
        lines.insert(187+ind+length1+length2+length3+length4,"    %s"%line)     #注意设定进程的空格以及行数
    s=''.join(lines)
    f1=open('main.py','w',encoding='utf-8')
    f1.write(s)
    f1.close()