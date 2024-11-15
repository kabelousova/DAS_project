from os import walk
import os.path as p
from time import ctime, strftime, strptime


def create_metafile(path, files):
    # print(files)
    global freq
    file_list_dat = list(filter(lambda x: True if x.endswith(".dat") else False, files))
    print(file_list_dat)
    for file in file_list_dat:
        try:
            freq = file.split("_")[4]
        except Exception as err:
            print("Error", err)
        print(file)
        print(path + file[:-4] + ".txt")
        with open(path + file[:-4] + ".txt", "w") as f:
            print("[Info]", file=f)
            print("Application=ADCLab SE [2.0.15.15]", file=f)
            print("BinaryFileOriginalPath=" + path + file + "", file=f)
            m_ti = p.getctime(path + file)
            t_obj = strptime(ctime(m_ti))
            T_stamp = strftime("%d.%m.%Y - %H:%M:%S", t_obj)
            print("DateTime=" + T_stamp + "", file=f)
            print("FullDeviceName=LAn10-12USB-U (100MHz 12 bit 4MB) ADC", file=f)
            print("Year=" + strftime("%Y", t_obj) + "", file=f)
            print("Month=" + strftime("%m", t_obj) + "", file=f)
            print("Day=" + strftime("%d", t_obj) + "", file=f)
            print("Hour=" + strftime("%H", t_obj) + "", file=f)
            print("Minute=" + strftime("%M", t_obj) + "", file=f)
            print("Second=" + strftime("%S", t_obj) + "", file=f)
            print("[Signal]", file=f)
            print("SampleSizeBytes=2" + "", file=f)
            print("IsSigned=true", file=f)
            print("AnalogDataBits=12", file=f)
            print("AnalogDataLSB=4", file=f)
            print("DigigalDataBits=0", file=f)
            print("DigigalDataLSB=0", file=f)

            print("SamplingRatePerChannel=" + freq[:-2] + "", file=f)
            print("MaxInputRange=2", file=f)
            print("ChannelNumberInFile=1", file=f)
            print("ChannelNumberDevice=2", file=f)
            print("[Channel_0]", file=f)
            print("Used=true", file=f)
            print("Gain=1", file=f)
            print("[Channel_1]", file=f)
            print("Used=false", file=f)
            print("Gain=1", file=f)


# create_metafile("D:\\Download\\main\\savedat\\", ["test4_12вцй_wdw_1000Hz.dat"])
for root, dirs, files in walk("D:\\WhyNotFreeNames\\Work\\DAS(ПИШ)\\Ключи_28.10.2024\\DAS+DVS_Ключи\\DAS_512_points\\2024-10-28_17-15-51\\"):
    create_metafile(root, files)
