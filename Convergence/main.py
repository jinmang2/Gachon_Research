# import library
import time
from convergence import *
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 오늘 날짜 받아오기
    now = time.localtime()
    today = '{}{:02}{:02}'.format(*now[:3])

    # receive filename and income, region variable name
    read_directory = input('파일을 가져올 디렉토리 주소를 입력해주세요 : ')
    user_input = input('확장자를 포함한 파일이름을 입력해주세요 : ')
    # filename = './input_data/' + user_input
    filename = read_directory + '/' + user_input
    income = input('\nincome 변수가 있다면 이름을 입력해주세요.\n없으면 Enter를 누르세요 : ')
    region = input('\nregion 변수가 있다면 이름을 입력해주세요.\n없으면 Enter를 누르세요 : ')
    to_directory = input('\n파일을 저장할 디렉토리 주소를 입력해주세요.\n없으면 Enter를 누르세요 : ')
    if to_directory == '':
        './table/'
    print('\n')
    # Read file and Start ConvergenceAnalysis
    df = load_excel(filename)

    if df is not None:
        CA = ConvergenceAnalysis(df, income=income, region=region)

        if CA.error == False:
            # Do Statistical Test
            CA.t_test()
            CA.chi_square_test()
            if CA.error == False:

                measure_table = CA.get_measure_table()
                index_table = CA.get_index_table()

                file, ets = user_input.split('.')
                _measure = file + '_measure_' + today + '.xlsx'
                _index = file + '_index_' + today + '.xlsx'

                try:
                    measure_table.to_excel(to_directory+'/'+_measure)
                    index_table.to_excel(to_directory+'/'+_index)
                    print('분석 종료. 지정한 폴더 안에 결과가 저장되었습니다.')
                except:
                    measure_table.to_excel('./table/'+_measure)
                    index_table.to_excel('./table/'+_index)
                    print('분석 종료. Table 폴더 안에 결과가 저장되었습니다.')
