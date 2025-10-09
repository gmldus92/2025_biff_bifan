import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity       
import urllib.parse 
import plotly.express as px


# =========================================================================
# 📌 1. 파일 경로 설정: (사용자님의 절대 경로 유지)
# =========================================================================
# FILE_PATH = '/Users/heeyeoncha/Desktop/영화제/list_total2.csv' 
FILE_PATH = 'list_total2.csv'


# =========================================================================
# 💡 2. 데이터 로딩 및 초기 필터링 함수 
# =========================================================================

@st.cache_data # 데이터 로딩 및 처리를 캐싱하여 앱 성능 향상
def load_data(file_path): 
    """
    CSV 파일을 읽고, 'festival' 컬럼을 기준으로 'Bucheon'과 'Busan' 데이터를 분리하는 함수.
    """
    required_cols = ['title', 'program_note', 'tag', 'country', 'festival']
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8') 
    except FileNotFoundError:
        st.error(f"❌ 오류: 지정된 파일 '{file_path}'을(를) 찾을 수 없습니다. 경로를 확인해 주세요.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ 데이터 로딩 중 오류 발생. (인코딩 문제일 수 있음): {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not all(col in df.columns for col in required_cols):
        st.error("🚨 데이터프레임에 필요한 컬럼('title', 'program_note', 'tag', 'country', 'festival') 중 일부가 누락되었습니다.")
        st.write("감지된 컬럼:", df.columns.tolist())
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bucheon_df = df[df['festival'] == 'Bucheon'].copy()
    busan_df = df[df['festival'] == 'Busan'].copy()

    return df, bucheon_df, busan_df


# =========================================================================
# 🎯 3. 추천 모델 함수 정의
# =========================================================================

@st.cache_data
def get_recommendations(df, sim_scores_array, top_n=5, exclude_self=False): 
    """
    특정 영화와의 유사도 점수 배열을 기준으로 추천 영화를 추출하는 함수.
    """
    sim_scores = list(enumerate(sim_scores_array[0])) 
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    if exclude_self:
        sim_scores = sim_scores[1:top_n + 1] 
    else:
        sim_scores = sim_scores[:top_n]
        

    movie_indices = [i[0] for i in sim_scores]
    
    recommended_movies = df.iloc[movie_indices].copy()
    recommended_movies['similarity'] = [i[1] for i in sim_scores]
    recommended_movies['tag'] = recommended_movies['tag'].fillna('')
    
    # --- [추가 로직: 왓챠피디아 링크 생성] ---
    BASE_URL = "https://pedia.watcha.com/ko-KR/search?query="
    recommended_movies['Watcha Link'] = recommended_movies['title'].apply(
        lambda title: BASE_URL + urllib.parse.quote(title)
    )
    # -------------------------------------
    
    return recommended_movies[['title', 'country', 'tag', 'similarity', 'Watcha Link']]


# ---------------------------------------------------------------------
# 📚 확장된 한국어 불용어 목록 적용 (TF-IDF에 사용)
# ---------------------------------------------------------------------
KOREAN_STOP_WORDS = [
    '아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을', '를', '에', '에게', '께', '에서', '으로서', '로', '와', '과', '더러', '까지', '까지도', '마저', '조차', '차', '부터', '대로', '만큼', '좇아', '처럼', '에', '로써', '갖고', '가지고', '그리고', '그렇지', '그렇다면', '하지만', '그러나', '그러니', '다시', '또', '이', '그', '저', '것', '수', '등', '등등', '때', '곳', '바', '대로', '로', '중', '없이', '뿐', '채', '줄', '및', '자', '이', '그', '이것', '그것', '저것', '이후', '대해', '관해', '대하여', '관하여', '때문', '으로부터', '만', '마저', '조차', '차', '뿐만', '같이', '처럼', '만큼', '따위', '이자', '하자', '더욱', '더군다나', '하물며', '게다가', '비록', '설령', '가령', '하더라도', '할지라도', '일지라도', '치더라도', '밖엔', '어찌', '하여', '든', '할', '바에야', '뿐이라', '뿐이랴', '기껏', '이만', '저만', '총', '연', '고작', '이래', '여사', '함께', '같이', '더불어', '도', '마저', '조차', '이', '그', '저', '각각', '각', '서로', '번갈아', '가며', '여기', '저기', '어디', '무엇', '언제', '누구', '남', '여', '별', '다른', '어느', '아무', '모두', '죄다', '전부', '도리어', '불과', '무론', '물론', '또한', '특히', '비단', '단지', '다만', '그저', '오직', '아무튼', '모쪼록', '부디', '여전히', '아직', '덜', '이제', '방금', '곧', '가까스로', '겨우', '다', '영', '훨씬', '얼마나', '정말', '참', '설마', '혹시', '다만', '허나', '한편', '이와', '같이', '따라서', '그러나', '그러므로', '대신', '혹은', '또는', '비슷하게', '틀림없이', '아니', '다시말하면', '좀', '잠깐', '잠시', '가량', '하마터면', '이밖에', '도대체', '차라리', '주로', '가령', '막론하고', '저마다', '한결', '하물며', '그나마', '하물며', '곧', '가까스로', '겨우', '다', '영', '훨씬', '얼마나', '정말', '참', '설마', '혹시', '다만', '허나', '한편', '이와', '같이', '따라서', '그러므로', '대신', '혹은', '또는', '비슷하게', '틀림없이', '아니', '다시말하면', '좀', '잠깐', '잠시', '가량', '하마터면', '이밖에', '도대체', '차라리', '주로', '가령', '막론하고', '저마다', '한결', '하물며', '그나마', '하물며'
]
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# 🎯 4. Feature Engineering 함수 정의
# ---------------------------------------------------------------------

def create_combined_features(row):
    """Program Note를 포함하고, tag 가중치를 8배, country 가중치를 2배로 조정"""
    
    tag = row['tag_for_features'] 
    country = row['country']
    program_note = row['program_note']
    
    # 장르(tag) 가중치를 8배로 상향
    tag_weighted = ' '.join([tag] * 8)
    # 국가(country) 가중치를 2배로 하향
    country_weighted = ' '.join([country] * 2)
    
    return f"{program_note} {tag_weighted} {country_weighted}"


# ---------------------------------------------------------------------
# ⚙️ 5. 데이터 로드 및 TF-IDF 계산 실행 블록
# ---------------------------------------------------------------------

# 데이터 로드
total_df, bucheon_df, busan_df = load_data(FILE_PATH)

# TF-IDF 변수 초기화
tfidf = None
total_tfidf_matrix = None
bucheon_tfidf_matrix = None
busan_tfidf_matrix = None # 🚨 초기화 추가

if not total_df.empty:
    # 인덱스 재설정
    total_df = total_df.reset_index(drop=True) 
    bucheon_df = bucheon_df.reset_index(drop=True) 
    busan_df = busan_df.reset_index(drop=True) 

    # 제외할 태그 키워드 리스트
    EXCLUDE_KEYWORDS = [
        '영화제', '칸', '베니스', '부산', '로카르노', '베를린', '선댄스', 
        '아카데미', '골든글로브', '상', '경쟁', '부문', '섹션', '프레젠테이션',
    ]

    # NaN 값 빈 문자열로 대체 및 태그 분리/정규화
    for df in [total_df, bucheon_df, busan_df]:
        
        # 1. 'tag' 컬럼 정제 및 UI 표시용 준비
        df['tag'] = df['tag'].astype(str).fillna('')
        
        # 2. 1차 정제: 구분자를 공백으로 통일하여 패턴 인식을 용이하게 함 (UI용 tag)
        df['tag'] = df['tag'].str.replace(r'[/\,]', ' ', regex=True) # 🚨 SyntaxWarning 수정: r'' 사용
        df['tag'] = df['tag'].str.replace(r'\s+', ' ', regex=True).str.strip() # 🚨 SyntaxWarning 수정: r'' 사용
        
        # 3. 키워드 기반 수상 태그 제거 로직 (UI용 tag)
        for keyword in EXCLUDE_KEYWORDS:
            df['tag'] = df['tag'].str.replace(
                rf'\b[^ ]*{keyword}[^ ]*\b', 
                '', 
                regex=True
            )

        # 4. 2차 후처리: 제거 후 남은 공백 정리 및 UI용 쉼표로 변환
        df['tag'] = df['tag'].str.replace(r'\s+', ' ', regex=True).str.strip() # 🚨 SyntaxWarning 수정: r'' 사용
        df['tag'] = df['tag'].str.replace(' ', ', ', regex=False) 

        # -----------------------------------------------------------------
        
        # 5. 'tag' 컬럼을 복사하여 특징 추출용 컬럼 생성
        df['tag_for_features'] = df['tag'].copy() 
        
        # 6. 'program_note'와 'country' 처리 (NaN 및 'nan' 문자열 제거)
        for col in ['program_note', 'country']:
            df[col] = df[col].astype(str).fillna('')
            df[col] = df[col].str.replace('nan', '', regex=False)
        
        # 7. 장르 분리 및 정규화 로직 (tag_for_features에 적용)
        # 쉼표를 다시 공백으로 변환하여 TF-IDF 토큰화 준비
        df['tag_for_features'] = df['tag_for_features'].str.replace('nan', '', regex=False)
        df['tag_for_features'] = df['tag_for_features'].str.replace(r'\,', ' ', regex=True) # 🚨 SyntaxWarning 수정: r'' 사용
        df['tag_for_features'] = df['tag_for_features'].str.replace(r'\s+', ' ', regex=True).str.strip() # 🚨 SyntaxWarning 수정: r'' 사용


    # combined_features 컬럼 생성
    total_df['combined_features'] = total_df.apply(create_combined_features, axis=1)
    bucheon_df['combined_features'] = bucheon_df.apply(create_combined_features, axis=1)
    busan_df['combined_features'] = busan_df.apply(create_combined_features, axis=1)

    # TF-IDF 모델 Fit & Transform
    tfidf = TfidfVectorizer(stop_words=KOREAN_STOP_WORDS)
    tfidf.fit(total_df['combined_features'])
    
    total_tfidf_matrix = tfidf.transform(total_df['combined_features'])
    bucheon_tfidf_matrix = tfidf.transform(bucheon_df['combined_features'])
    busan_tfidf_matrix = tfidf.transform(busan_df['combined_features']) 
    
    # ---------------------------------------------------------------------
    # 🚨 그래프 섹션에 사용될 데이터프레임이 비어있지 않다면,
    # Main 실행 블록에서 그래프 함수를 호출할 수 있도록 준비
    # ---------------------------------------------------------------------

# =========================================================================
# 📊 6. 그래프 섹션 함수 정의 (파일 상단에 위치)
# =========================================================================

# 1. 공통 기능: 비율 문자열을 float으로 변환하는 함수
def clean_ratio(s):
    """비율 문자열 (%, - 포함)을 숫자로 변환합니다."""
    if pd.isna(s) or s == '-':
        return 0.0
    return float(str(s).strip().replace('%', ''))

# 2-1. BIFF vs BIFAN 비교 분석 데이터 정의 및 전처리
@st.cache_data
def prepare_data_biff_bifan_comp():
    """BIFF와 BIFAN의 국가별 비율 비교 데이터를 전처리합니다."""
    # (데이터 정의 및 전처리 로직 유지)
    data = [
        {'Country': 'korea', 'BIFF_ratio': '17.5%', 'Busan_ratio': '41.9%'},
        {'Country': 'france', 'BIFF_ratio': '15.0%', 'Busan_ratio': '5.1%'},
        {'Country': 'unitedstates', 'BIFF_ratio': '7.0%', 'Busan_ratio': '-'},
        {'Country': 'japan', 'BIFF_ratio': '5.8%', 'Busan_ratio': '6.0%'},
        {'Country': 'italy', 'BIFF_ratio': '5.1%', 'Busan_ratio': '0.9%'},
        {'Country': 'germany', 'BIFF_ratio': '4.9%', 'Busan_ratio': '0.9%'},
        {'Country': 'belgium', 'BIFF_ratio': '2.7%', 'Busan_ratio': '2.6%'},
        {'Country': 'china', 'BIFF_ratio': '2.7%', 'Busan_ratio': '3.4%'},
        {'Country': 'india', 'BIFF_ratio': '2.4%', 'Busan_ratio': '0.9%'},
        {'Country': 'taiwan', 'BIFF_ratio': '2.2%', 'Busan_ratio': '2.1%'},
        {'Country': 'hongkong,china', 'BIFF_ratio': '1.9%', 'Busan_ratio': '-'},
        {'Country': 'indonesia', 'BIFF_ratio': '1.7%', 'Busan_ratio': '0.4%'},
        {'Country': 'netherlands', 'BIFF_ratio': '1.7%', 'Busan_ratio': '0.9%'},
        {'Country': 'singapore', 'BIFF_ratio': '1.5%', 'Busan_ratio': '1.3%'},
        {'Country': 'spain', 'BIFF_ratio': '1.5%', 'Busan_ratio': '1.7%'},
        {'Country': 'unitedkingdom', 'BIFF_ratio': '1.5%', 'Busan_ratio': '-'},
        {'Country': 'denmark', 'BIFF_ratio': '1.2%', 'Busan_ratio': '1.3%'}
    ]
    
    df = pd.DataFrame(data)
    
    df['BIFF_ratio_val'] = df['BIFF_ratio'].apply(clean_ratio)
    df['Busan_ratio_val'] = df['Busan_ratio'].apply(clean_ratio)
    
    # 그래프를 위한 Long-format 변환
    df_long = pd.melt(df, id_vars=['Country'], value_vars=['BIFF_ratio_val', 'Busan_ratio_val'],
                      var_name='Festival', value_name='Ratio')
    
    df_long['Festival'] = df_long['Festival'].replace({
        'BIFF_ratio_val': '부산국제영화제 (BIFF)',
        'Busan_ratio_val': '부천국제판타스틱영화제 (BIFAN)'
    })
    
    # Country 이름 정리
    df_long['Country'] = df_long['Country'].replace({
        'unitedstates': 'USA', 
        'unitedkingdom': 'UK',
        'hongkong,china': 'Hongkong'
    }).str.title().replace('Usa', 'USA').replace('Uk', 'UK')
    
    return df_long

# 2-2. BIFAN 단독 분석 데이터 정의 및 전처리
@st.cache_data
def prepare_data_bifan_single():
    """BIFAN 국가별 단독 비율 데이터를 전처리합니다."""
    # (데이터 정의 및 전처리 로직 유지)
    data = [
        {'Country': 'korea', 'Ratio': '41.9%'},
        {'Country': 'usa', 'Ratio': '9.4%'},
        {'Country': 'japan', 'Ratio': '6.0%'},
        {'Country': 'france', 'Ratio': '5.1%'},
        {'Country': 'china', 'Ratio': '3.4%'},
        {'Country': 'canada', 'Ratio': '3.4%'},
        {'Country': 'belgium', 'Ratio': '2.6%'},
        {'Country': 'taiwan', 'Ratio': '2.1%'},
        {'Country': 'sweden', 'Ratio': '1.7%'},
        {'Country': 'spain', 'Ratio': '1.7%'},
        {'Country': 'uk', 'Ratio': '1.3%'},
        {'Country': 'denmark', 'Ratio': '1.3%'},
        {'Country': 'singapore', 'Ratio': '1.3%'},
        {'Country': 'australia', 'Ratio': '1.3%'},
        {'Country': 'hongkong', 'Ratio': '1.3%'},
        {'Country': 'finland', 'Ratio': '0.9%'},
        {'Country': 'norway', 'Ratio': '0.9%'},
    ]
    
    df = pd.DataFrame(data)
    df['Ratio_val'] = df['Ratio'].apply(clean_ratio)
    df['Country'] = df['Country'].str.title().replace('Usa', 'USA').replace('Uk', 'UK')
    
    return df

# 2-3. 국가별 비교 그래프 렌더링 함수
def render_country_chart_comp_top10():
    """BIFF vs BIFAN 상위 10개 국가 비교 그래프 렌더링"""
    
    df_chart = prepare_data_biff_bifan_comp()
    N_TOP = 10 
    
    top_countries = df_chart[df_chart['Festival'] == '부산국제영화제 (BIFF)']\
        .sort_values(by='Ratio', ascending=False)['Country'].head(N_TOP).tolist()
    
    df_filtered = df_chart[df_chart['Country'].isin(top_countries)]
    
    st.subheader("🌏 BIFF vs. BIFAN 출품 국가")
    st.markdown(f"BIFF 비율 기준 상위 **{N_TOP}개 국가**의 두 영화제 참여 비율을 비교합니다.")

    fig = px.bar(df_filtered, x='Country', y='Ratio', color='Festival', barmode='group', 
                 title='BIFF 기준 상위 10개 국가의 영화제 참여 비율',
                 labels={'Country': '국가', 'Ratio': '참여 비율 (%)'}, height=550)
    
    fig.update_traces(hovertemplate='<b>%{y:.2f}%</b><extra></extra>') 
    fig.update_layout(xaxis={'categoryorder': 'total descending'}, legend_title_text='영화제', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # 그래프 하단 한 줄 요약
    summary_text = (
        "BIFF는 국내를 비롯한 프랑스,이탈리아, 독일 등 유럽권과 미국, 일본 등의 출품작이 균형있게 분포되어 있습니다.<br>"
        "BIFAN과 동일 국가를 놓고 비교했을 때, 아시아권 영화 비율은 조금 낮을 수 있지만 더 다양한 국가에서 출품하고 있음을 알 수 있습니다.<br>"
        "국내 최대 국제영화제인만큼 다양한 국가의 다양한 작품을 만나볼 수 있을 것으로 기대됩니다."
    )
    st.markdown(f'<p style="font-size: 16px; color: #FFC107; border-left: 5px solid #FFC107; padding-left: 10px;">{summary_text}</p>', 
                unsafe_allow_html=True)

# 2-4. BIFAN 단독 그래프 렌더링 함수
def render_country_chart_bifan_top10():
    """BIFAN 상위 10개 국가의 단독 그래프 렌더링"""
    
    df_chart = prepare_data_bifan_single()
    N_TOP = 10 
    
    df_filtered = df_chart.sort_values(by='Ratio_val', ascending=False).head(N_TOP).copy()
    
    st.subheader(f"👻 BIFAN (부천국제판타스틱영화제) 상위 {N_TOP}개 국가 참여 비율")
    st.markdown("BIFAN 참여 비율을 기준으로 정렬된 막대 그래프입니다.")

    fig = px.bar(df_filtered, x='Country', y='Ratio_val', 
                 title=f'BIFAN 참여 비율 상위 {N_TOP}개 국가',
                 labels={'Country': '국가', 'Ratio_val': '참여 비율 (%)'},
                 height=550, color='Ratio_val', color_continuous_scale=px.colors.sequential.Teal)
    
    fig.update_traces(hovertemplate='<b>%{y:.2f}%</b><extra></extra>')
    fig.update_layout(xaxis={'categoryorder': 'total descending'}, yaxis_title='참여 비율 (%)',
                      title_x=0.5, coloraxis_colorbar_title='비율')
    st.plotly_chart(fig, use_container_width=True)
    
    # 그래프 하단 한 줄 요약
    summary_text = (
        "BIFAN은 압도적으로 국내 작품이 많이 출품했는데, 국내 단편 섹션이 따로 있어 작품 수가 많을 수 있습니다.<br>"
        "다양한 국가에서 출품했지만, 가장 돋보이는 국가는 한국이므로 유수한 국내 작품을 가장 많이 만나 볼 수 있습니다.<br>"
        "국내 최대 장르영화제인 만큼 장르영화에서 두각을 나타내는 국가의 작품과 국내 작품 모두 만날 수 있는 기회입니다. "
    )
    st.markdown(f'<p style="font-size: 16px; color: #FFC107; border-left: 5px solid #FFC107; padding-left: 10px;">{summary_text}</p>', 
                unsafe_allow_html=True)

# 3-1. BIFF 장르별 분포 데이터 정의 및 전처리
@st.cache_data
def prepare_data_biff_genres():
    """BIFF 장르별 분포 (추정) 데이터를 전처리합니다."""
    # (데이터 정의 및 전처리 로직 유지)
    data_text = """
    genre가족/아동659.76%
    genre성장영화/청춘619.16%
    genre여성497.36%
    genre심리/미스터리/서스펜스/스릴러487.21%
    genre인권/노동/사회456.76%
    genre사랑/연애/로맨스426.31%
    genre실화바탕324.80%
    genre코미디/유머/블랙코미디/풍자294.35%
    genre범죄/폭력274.05%
    genre여행/로드무비253.75%
    """
    lines = data_text.strip().split('\n')
    processed_data = []
    
    for line in lines:
        line = line.replace('genre', '', 1).strip()
        ratio_str = line[-5:] 
        ratio_val = clean_ratio(ratio_str)
        
        # 비율 문자열 앞에 있는 카운트(cnt) 제거 로직
        genre_name = line[:-5]
        while genre_name and genre_name[-1].isdigit():
             genre_name = genre_name[:-1]
        
        processed_data.append({'Genre': genre_name.strip(), 'Ratio_val': ratio_val})
        
    return pd.DataFrame(processed_data)

# 3-2. BIFAN 장르별 분포 데이터 정의 및 전처리
@st.cache_data
def prepare_data_bifan_genres():
    """BIFAN 장르별 분포 데이터를 전처리합니다."""
    # (데이터 정의 및 전처리 로직 유지)
    data = [
        {'Genre': '가족', 'Ratio': '7.58%'},
        {'Genre': '스릴러', 'Ratio': '6.31%'},
        {'Genre': '로맨스', 'Ratio': '5.30%'},
        {'Genre': '감동', 'Ratio': '5.05%'},
        {'Genre': '범죄/누아르', 'Ratio': '4.80%'},
        {'Genre': '액션', 'Ratio': '4.29%'},
        {'Genre': '블랙코미디', 'Ratio': '4.29%'},
        {'Genre': '코미디', 'Ratio': '4.29%'},
        {'Genre': 'SF', 'Ratio': '3.79%'},
        {'Genre': '하드고어', 'Ratio': '3.79%'},
    ]
    df = pd.DataFrame(data)
    df['Ratio_val'] = df['Ratio'].apply(clean_ratio)
    
    return df

# 3-3. 장르별 분포 그래프 렌더링 공통 함수
def render_genre_chart(df, title, festival_name, summary_color, summary_text):
    """장르별 분포 막대 그래프를 그리는 공통 함수"""
    
    fig = px.bar(df, x='Genre', y='Ratio_val', 
                 title=f'{festival_name} 장르별 영화 분포 (상위 10개)',
                 labels={'Genre': '장르', 'Ratio_val': '비율 (%)'},
                 height=500, color='Ratio_val', color_continuous_scale=px.colors.sequential.Blues)
    
    fig.update_traces(hovertemplate='<b>%{y:.2f}%</b><extra></extra>') 
    fig.update_layout(xaxis={'categoryorder': 'total descending'}, yaxis_title='장르별 비중 (%)',
                      title_x=0.5, coloraxis_colorbar_title='비율')
    
    st.markdown(f"**{festival_name}**의 주요 장르별 분포입니다. 막대에 마우스를 올려보세요.")
    st.plotly_chart(fig, use_container_width=True)
    
    # 그래프 하단 한 줄 요약
    st.markdown(f'<p style="font-size: 16px; color: {summary_color}; border-left: 5px solid {summary_color}; padding-left: 10px;">{summary_text}</p>', 
                unsafe_allow_html=True)


# =============================================================
# 7. Streamlit 앱 메인 실행 블록 (전체 앱 구성)
# =============================================================

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="영화제 통합 분석 대시보드")
    
    # -------------------------------------------------------------
    # 7-1. 추천 시스템 UI (기존 5. Streamlit 앱 인터페이스 구성 내용)
    # -------------------------------------------------------------
    
    st.title("🎬 2025 BIFF 영화 데이터로 보는 BIFAN 과의 차이점")

    if total_df.empty:
        st.warning("데이터 로드에 실패하여 앱을 더 이상 진행할 수 없습니다. 파일 경로 및 내용을 확인해주세요.")
        st.stop() 

    st.success(f"총 **{len(total_df)}**개의 영화 데이터를 로드했습니다.")
    st.markdown(f"**25' BIFF** 영화: {len(busan_df)}개 | **25' BIFAN** 영화: {len(bucheon_df)}개")

    st.markdown("---")

    if not busan_df.empty and total_tfidf_matrix is not None:
        
        st.subheader("👀 이번 2025 BIFF에서 선택한 영화는 무엇인가요?")
        
        movie_titles = sorted(busan_df['title'].unique().tolist())
        
        selected_movie = st.selectbox(
            "추천 기준이 될 **BIFF 영화**를 선택하세요:",
            options=movie_titles,
            index=0 
        )

        st.markdown("---")
        st.info(f"BIFF 선택 영화: **{selected_movie}**")

        selected_data = busan_df[busan_df['title'] == selected_movie].iloc[0]
        selected_features = selected_data['combined_features'] 
        
        with st.expander(f"'{selected_movie}'의 상세 정보 보기"):
        
        # 🚨 [추가] 트레일러 URL 컬럼을 확인하여 영상 임베딩
        # 'trailer_url' 컬럼이 존재하고 값이 비어있지 않다면 영상을 표시합니다.
        # (컬럼 이름이 'trailer_url'이 아닐 경우, 실제 컬럼 이름으로 바꿔주세요.)
         if 'trailer_url' in selected_data and selected_data['trailer_url']:
            st.subheader("🎬 트레일러")
            # st.video()를 사용하여 유튜브 URL 임베딩
            st.video(selected_data['trailer_url'], width=400)
        
            st.write("**태그 (장르 대용):**", selected_data['tag'])
            st.write("**국가:**", selected_data['country'])
            st.write("**프로그램 노트:**", selected_data['program_note'])
        
            st.markdown("---")
        
        if not selected_features.strip(): 
            st.warning(f"선택하신 영화 '{selected_movie}'는 **복합 정보(프로그램 노트/태그/국가)가 모두 부족**하여 유사도 기반 추천을 생성할 수 없습니다. 다른 영화를 선택해 주세요.")
        else:
            selected_movie_vec = tfidf.transform([selected_features])
            
            # 뷰 1: 부산/부천 (Total List)에서 추천
            st.subheader("🌟 2025 BIFF/BIFAN에서는 이 영화와 비슷해요")
            total_sim_scores = cosine_similarity(selected_movie_vec, total_tfidf_matrix)

            total_recommendations = get_recommendations(
                total_df, 
                total_sim_scores, 
                top_n=5,
                exclude_self=True 
            )

            if not total_recommendations.empty:
                st.dataframe(total_recommendations,
                    column_config={
                        "similarity": st.column_config.ProgressColumn("유사도", format="%.2f", min_value=0.0, max_value=1.0),
                        "Watcha Link": st.column_config.LinkColumn("왓챠 링크", display_text="바로가기"),
                    },
                    use_container_width=True
                )
            else:
                st.warning("전체 목록에서 추천 결과를 찾지 못했습니다.")
                
            st.markdown("---")

            # 뷰 2: 부천 영화제 (Bucheon List)에서 추천
            st.subheader("🌟 2025 BIFAN에서는 이 영화와 비슷해요")

            if bucheon_tfidf_matrix is not None and bucheon_df.shape[0] > 0:
                bucheon_sim_scores = cosine_similarity(selected_movie_vec, bucheon_tfidf_matrix)

                bucheon_recommendations = get_recommendations(
                    bucheon_df, 
                    bucheon_sim_scores, 
                    top_n=5,
                    exclude_self=False 
                )

                if not bucheon_recommendations.empty:
                    st.dataframe(bucheon_recommendations,
                        column_config={
                            "similarity": st.column_config.ProgressColumn("유사도", format="%.2f", min_value=0.0, max_value=1.0),
                            "Watcha Link": st.column_config.LinkColumn("왓챠 링크", display_text="바로가기"),
                        },
                        use_container_width=True
                    )
                else:
                    st.warning("부천 영화제 목록에서 추천 결과를 찾지 못했습니다.")
            else:
                st.warning("부천 영화제 데이터가 없거나 모델 준비에 실패했습니다.")

    else:
        st.warning("Busan 영화 데이터가 존재하지 않거나, 추천 모델 준비에 실패했습니다.")
        
    st.markdown("---")

    # -------------------------------------------------------------
    # 7-2. 그래프 섹션 UI (새로 추가된 내용)
    # -------------------------------------------------------------

    st.header("📊 데이터로 보는 BIFF VS BIFAN")
    st.markdown(f"#### 아시아 최대 규모 국제영화제 BIFF와 아시아 최대 장르영화제 BIFAN은 어떤 차이가 있을까?")
    
    # 탭 3개를 생성하여 분석 주제별로 분리
    tab_country_comp, tab_genre = st.tabs(
        ["✔️ 출품 국가별 비교",  "✔️ 영화 장르별 비교"]
    )

    # === 탭 1: 국가별 비교 분석 ===
    with tab_country_comp:
        # 1. BIFF vs BIFAN 비교 분석
        render_country_chart_comp_top10() 
        
        st.markdown("---")
        
        # 🚨 2. BIFAN 국가별 단독 분석을 바로 하단에 추가
        render_country_chart_bifan_top10() 

    # === 탭 2: 장르별 분석 (두 그래프를 컬럼으로 나누어 비교) ===
    with tab_genre:
        st.subheader("🎭 25' BIFF vs. BIFAN 장르별 분포 비교")
        st.markdown("---")
        
        df_biff_genre = prepare_data_biff_genres()
        df_bifan_genre = prepare_data_bifan_genres()
        
        col_biff, col_bifan = st.columns(2)
        
        biff_summary_text = "2025년 부산국제영화제에 출품한 영화는 가족/아동(9.8%)), 성장영화/청춘(9.6%)가 상위를 차지하며 다양한 연령층의 관객들이 좋아할만한 장르의 영화로 구성되어 있습니다.또한 여성, 인권/노동/사회 영화의 비중도 높아 전반적으로 휴머니즘의 색챌르 띄고 있습니다. 심리/미스터리/서프펜스 장르 역시 4위를 차지하며 박진감 넘치는 작품도 다수 포진되어 있을 것으로 기대됩니다."
        
        with col_biff:
            render_genre_chart(
                df=df_biff_genre, 
                title="BIFF 장르(tag) 분포", 
                festival_name="BIFF (부산국제영화제)",
                summary_color="#FFC107",
                summary_text=f"{biff_summary_text}"
            )
            
        bifan_summary_text = "2025년 부천 영화제는 장르영화제인 만큼 미스터리, 호러 등의 장르가 다수를 차지할 것이라고 생각했는데요, 가족 장르의 영화가 약 7.6%로 1위를 차지하고, 로맨스, 감동 장르 역시 3,4위를 차지하여 높은 비율을 보여줬습니다. '저 세상 패밀리', '메리 고 라운드' 등의 섹션의 영향으로 예상됩니다. 스릴러가 2위를 차지하고, 범죄누아르, SF, 하드고어 등 BIFF에서는 잘 보이지 않던 장르가 상위를 차지한 점이 BIFAN 장르영화제의 특징이라고 해석할 수 있을 것 같습니다. "
            
        with col_bifan:
            render_genre_chart(
                df=df_bifan_genre, 
                title="BIFAN 장르(tag) 분포", 
                festival_name="BIFAN (부천국제판타스틱영화제)",
                summary_color="#FFC107",
                summary_text=f"{bifan_summary_text}"
            )