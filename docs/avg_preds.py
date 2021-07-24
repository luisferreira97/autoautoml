import pandas as pd
from statistics import mean

df = pd.read_csv("tempos.csv", sep=";")

df["tempo"] = df["tempo"].astype(str)


final_df = pd.DataFrame(
    columns=['dataset', 'categoria', 'ferramenta', 'tempo_medio'])

for x in range(0, 132):
    start_index = x * 10
    end_index = start_index + 10
    print("\n")
    time_list = []
    nova_linha = {
        "dataset": df["dataset"].iloc[start_index],
        "categoria": df["categoria"].iloc[start_index],
        "ferramenta": df["ferramenta"].iloc[start_index],
        "tempo_medio": ""
    }
    for y in range(start_index, end_index):
        valor_tempo = df["tempo"].iloc[y]
        segundos = int(valor_tempo[0:2]) * 60 + int(valor_tempo[3:5])
        # print(segundos)
        time_list.append(segundos)
    # print(time_list)
    media_tempos = round(mean(time_list))
    # print(media_tempos)
    media_minutos = int(media_tempos / 60)
    if len(str(media_minutos)) == 1:
        media_minutos = "0" + str(media_minutos)
    media_segundos = media_tempos % 60
    if media_segundos < 10:
        media_segundos = "0" + str(media_segundos)
    media_string = str(media_minutos) + ":" + str(media_segundos)
    print(media_string)
    nova_linha["tempo_medio"] = media_string
    final_df = final_df.append(nova_linha, ignore_index=True)

print(final_df)
final_df.to_csv("tempos_medios.csv", sep=",", index=False)
