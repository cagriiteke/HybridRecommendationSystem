
#############################################
# PROJE: Hybrid Recommender System
#############################################

# user_id = 108170

#############################################
# Adım 1: Verinin Hazırlanması
#############################################
import pandas as pd

rating = pd.read_csv('RecommendationSystem/rating.csv')
movie = pd.read_csv('RecommendationSystem/movie.csv')

# movieleri ve puanlarını birleştirelim
df = movie.merge(rating, how="left", on="movieId")
df.head()

df.info()

# timestamp türü object olduğu için değiştirelim
df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d')
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df.head()


def create_user_movie_df():
    import pandas as pd
    # içinde 4 değişken olan parantez içi yılları seçiyor
    df['year_movie'] = df.title.str.extract('(\(\d\d\d\d\))', expand=False)
    # sonra bu parantez içindeki yılları yeni değişkene atıyor
    df['year_movie'] = df.year_movie.str.extract('(\d\d\d\d)', expand=False)
    # title değişkenindeki yılı seçip siliyoruz
    df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')
    # kalan boşlukları siliyoruz
    df['title'] = df['title'].apply(lambda x: x.strip())
    a = pd.DataFrame(df["title"].value_counts())
    # 1000'den küçük olan titleları seçip drop ediyoruz
    rare_movies = a[a["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df


user_movie_df = create_user_movie_df()

user_movie_df.head()

#############################################
# Adım 2: Öneri yapılacak kullanıcının izlediği filmlerin belirlenmesi
#############################################

random_user = 108170
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()


# kullanıcının izlediği filmleri bulmalıyız
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched[0:5]

len(movies_watched)
# 186

#############################################
# Adım 3: Aynı filmleri izleyen diğer kullanıcıların verisine ve id'lerine erişmek
#############################################
pd.set_option('display.max_columns', 5)

movies_watched_df = user_movie_df[movies_watched]

movies_watched_df.shape
# (138493, 186)

movies_watched_df.head()


#############################################
# Adım 4: Öneri yapılacak kullanıcı ile en benzer kullanıcıların belirlenmesi
#############################################

# userid ile en az %60 aynı filmi izleyen kullanıcılar ile userid'nin izlediği filmleri bir araya getiriyoruz

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      random_user_df[movies_watched]])

final_df.head()


# kullanıcıların korelasyonlarını inceleyelim
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()
#  user_id_1  user_id_2  corr
# 0   124860.0     9445.0  -1.0
# 1    78680.0   107571.0  -1.0
# 2    24424.0    78841.0  -1.0
# 3     5543.0    76841.0  -1.0
# 4   119366.0   124122.0  -1.0

# userid ile benzer kullanıcıların korelasyonu 0.65'ten eşit-büyük olanları alalım
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.head()
#      userId      corr
# 54   24218.0  0.993808
# 53   16839.0  0.991140
# 52  135521.0  0.970148
# 51   23164.0  0.933848
# 50   34699.0  0.902826

# rating tablosundan bu top users kullanıcıların filmlere verdikleri puanlar:
# Bu kullanıcılar userid ile benzer, hangi filme ne kadar oy vermişler bakalım:
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings.head()
#   userId      corr  movieId  rating
# 0  24218.0  0.993808       19     0.5
# 1  24218.0  0.993808      318     5.0
# 2  24218.0  0.993808      344     0.5
# 3  24218.0  0.993808      435     4.5
# 4  24218.0  0.993808      541     4.5

#############################################
# Adım 5: Weighted rating'lerin  hesaplanması
#############################################

# ratinglerde korelasyone göre düzeltme yapacağız.
# Çok benzer olanları ve ratingleri ağırlıklandırmış olduk.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

#  userId      corr  movieId  rating  weighted_rating
# 0  24218.0  0.993808       19     0.5         0.496904
# 1  24218.0  0.993808      318     5.0         4.969040
# 2  24218.0  0.993808      344     0.5         0.496904
# 3  24218.0  0.993808      435     4.5         4.472136
# 4  24218.0  0.993808      541     4.5         4.472136


#############################################
# Adım 6: Weighted average recommendation score'un hesaplanması ve ilk beş filmin tutulması
#############################################

# Filmler özelinde tekilleştirme yaparak benzer kullanıcıların ratingleri olmalı.
temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
temp.columns = ['sum_corr', 'sum_weighted_rating']

temp.head()
#        sum_corr  sum_weighted_rating
# movieId
# 1        10.695522            33.171936
# 2         4.397580            16.206339
# 3         1.342875             3.359913
# 5         1.336727             3.347617
# 6         4.849560            18.394512


# tavsiye için ortalama puan hesaplıyoruz
recommendation_df = pd.DataFrame()
recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
recommendation_df['movieId'] = temp.index
recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
recommendation_df


# en iyi kullanıcılar için film isimlerini movie veri setinden alıyoruz
movies_from_user_based_first = movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'])]
movies_from_user_based_first


movies_from_user_based = movies_from_user_based_first[0:5]
movies_from_user_based

#############################################
# Adım 7: İzlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
# 5 öneri user-based 5 öneri item-based olacak şekilde 10 öneri yapınız.
#############################################

lastmovie_df = pd.DataFrame()
lastmovie_df = df[df['userId'] == random_user].sort_values(["rating", "timestamp"], ascending=False)
lastmovie = "Wild at Heart"



def item_based_recommender(movie_name):
    # film umd'de yoksa önce ismi barındıran ilk filmi getir.
    # eger o da yoksa filmin isminin ilk iki harfini barındıran ilk filmi getir.
    if movie_name not in user_movie_df:
        # ismi barındıran ilk filmi getir.
        if [col for col in user_movie_df.columns if movie_name.capitalize() in col]:
            new_movie_name = [col for col in user_movie_df.columns if movie_name.capitalize() in col][0]
            movie = user_movie_df[new_movie_name]
            print(F"{movie_name}'i barındıran ilk  film: {new_movie_name}\n")
            print(F"{new_movie_name} için öneriler geliyor...\n")
            return user_movie_df.corrwith(movie).sort_values(ascending=False).head(6)
        # filmin ilk 2 harfini barındıran ilk filmi getir.
        else:
            new_movie_name = [col for col in user_movie_df.columns if col.startswith(movie_name.capitalize()[0:2])][0]
            movie = user_movie_df[new_movie_name]
            print(F"{movie_name}'nin ilk 2 harfini barındıran ilk film: {new_movie_name}\n")
            print(F"{new_movie_name} için öneriler geliyor...\n")
            return user_movie_df.corrwith(movie).sort_values(ascending=False).head(6)
    else:
        print(F"{movie_name} için öneriler geliyor...\n")
        movie = user_movie_df[movie_name]
        return user_movie_df.corrwith(movie).sort_values(ascending=False).head(6)


movies_from_item_based = item_based_recommender(lastmovie)
movies_from_item_based = movies_from_item_based[1:6]
movies_from_item_based


movies_from_item_based = movies_from_item_based.reset_index()

movies_from_user_based = movies_from_user_based.reset_index()

recommendations = pd.DataFrame()
recommendations["item_based"] = movies_from_item_based["title"]
recommendations["user_based"] = movies_from_user_based["title"]

recommendations

