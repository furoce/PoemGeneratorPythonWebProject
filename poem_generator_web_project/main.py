# # coding: UTF-8
# '''''''''''''''''''''''''''''''''''''''''''''''''''''
#    file name: main.py
#    create time: 2017年06月23日 星期五 16时41分54秒
#    author: Jipeng Huang
#    e-mail: huangjipengnju@gmail.com
#    github: https://github.com/hjptriplebee
# '''''''''''''''''''''''''''''''''''''''''''''''''''''
# from poem_generator_web_project.config import *
# import poem_generator_web_project.data as data
# import poem_generator_web_project.model as model
#
# def defineArgs():
#     """define args"""
#     parser = argparse.ArgumentParser(description = "Chinese_poem_generator.")
#     parser.add_argument("-m", "--mode", help = "select mode by 'train' or test or head",
#                         choices = ["train", "test", "head"], default = "test")
#     return parser.parse_args()
#
# if __name__ == "__main__":
#     args = defineArgs()
#     trainData = data.POEMS(trainPoems)
#     MCPangHu = model.MODEL(trainData)
#     if args.mode == "train":
#         MCPangHu.train()
#     else:
#         if args.mode == "test":
#             poems = MCPangHu.test()
#         else:
#             characters = input("please input chinese character:")
#             poems = MCPangHu.testHead(characters)