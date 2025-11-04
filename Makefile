# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: nasr <nasr@student.42.fr>                  +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/10/23 22:47:45 by nhanafi           #+#    #+#              #
#    Updated: 2025/11/04 23:53:28 by nasr             ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Compilers
CC = c++
NVCC = nvcc

# Compiler flags
CFLAGS = -std=c++20 -Wall -Werror -Wextra
NVCCFLAGS = -std=c++20 -Xcompiler -Wall,-Wextra
CUDA_LIBS = -lcudart -lcurand

# Source files
SRC = LogisticRegression DataLoader \
	  Matrix/constructors Matrix/functions Matrix/operations Matrix/property Matrix/vectorOperations

CUDA_SRC = cuMatrix/constructors cuMatrix/functions cuMatrix/operations cuMatrix/property cuMatrix/vectorOperations cuMatrix/kernels

INC = include

HEADER = include/DataLoader.h include/LogisticRegression.h include/Matrix/Matrix.h include/cuMatrix/MatrixCuda.h
ODIR = obj

OBJ = $(addprefix $(ODIR)/, $(SRC:=.o))
CUDA_OBJ = $(addprefix $(ODIR)/, $(CUDA_SRC:=.o))

BUILD = build

NAME = $(BUILD)/dslr.a
CUDA_NAME = $(BUILD)/dslr_cuda.a

all: cpu cuda

cpu: $(NAME)

cuda: $(CUDA_NAME)

both: $(NAME) $(CUDA_NAME)

$(NAME): $(OBJ)
	mkdir -p $(BUILD)
	ar rc $(NAME) $(OBJ)

$(CUDA_NAME): $(CUDA_OBJ)
	mkdir -p $(BUILD)
	ar rc $(CUDA_NAME) $(CUDA_OBJ)

$(ODIR)/%.o: src/%.cpp $(HEADER)
	@echo $<
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(INC)  -c $< -o $@

$(ODIR)/%.o: src/%.cu $(HEADER)
	@echo $<
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -I$(INC) -c $< -o $@

clean:
	rm -rf $(ODIR)

fclean: clean
	rm -rf $(NAME) $(CUDA_NAME) $(BUILD)

re: fclean all

.PHONY : re fclean clean all cuda cpu both