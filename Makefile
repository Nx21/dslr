# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: nasr <nasr@student.42.fr>                  +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/10/23 22:47:45 by nhanafi           #+#    #+#              #
#    Updated: 2025/10/07 23:27:40 by nasr             ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


CC = c++

CFLAGS = -std=c++20 -Wall -Werror -Wextra
SRC = LogisticRegression DataLoader \
	  Matrix/constructors Matrix/functions Matrix/operations Matrix/property Matrix/vectorOperations
INC = include

HEADER = include/DataLoader.h include/LogisticRegression.h include/Matrix.h
ODIR = obj

OBJ = $(addprefix $(ODIR)/, $(SRC:=.o))

BUILD = build

NAME = $(BUILD)/dslr.a

all: $(NAME)

$(NAME): $(OBJ)
	mkdir -p $(BUILD)
	ar rc $(NAME) $(OBJ)

$(ODIR)/%.o: src/%.cpp $(HEADER)
	@echo $<
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(INC)  -c $< -o $@

clean:
	rm -rf $(ODIR)

fclean: clean
	rm -rf $(NAME) $(BUILD)

re: fclean all

.PHONY : re fclean clean all