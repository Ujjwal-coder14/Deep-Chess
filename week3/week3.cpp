#include<chess.hpp>
using namespace chess;

const int INF = 1e9;

int evaluate(Board &board){
    // A simple evaluation function that counts material
    int score = 0;
    for(int i=0; i<64; i++){
        Piece piece = board.at(Square(i));
        int pieceValue = 0;
        if(piece == Piece::WHITEPAWN) pieceValue = 100;
        else if(piece == Piece::WHITEKNIGHT) pieceValue = 320;
        else if(piece == Piece::WHITEBISHOP) pieceValue = 320;
        else if(piece == Piece::WHITEROOK) pieceValue = 500;
        else if(piece == Piece::WHITEQUEEN) pieceValue = 900;
        else if(piece == Piece::WHITEKING) pieceValue = 20000;
        else if(piece == Piece::BLACKPAWN) pieceValue = -100;
        else if(piece == Piece::BLACKKNIGHT) pieceValue = -320;
        else if(piece == Piece::BLACKBISHOP) pieceValue = -320;
        else if(piece == Piece::BLACKROOK) pieceValue = -500;
        else if(piece == Piece::BLACKQUEEN) pieceValue = -900;
        else if(piece == Piece::BLACKKING) pieceValue = -20000;
        else pieceValue = 0;
        score += pieceValue;
    }
    return score;
}

std::pair<int, Move> minimax(Board &board, int depth, bool maximizing){
    Movelist moves;
    movegen::legalmoves(moves, board);

    if(moves.empty()){
        if(board.inCheck()){
            if(maximizing)
                return {-INF, Move()};
            else
                return {INF, Move()};
        }
        return {0, Move()};
    }

    if(depth == 0){
        return {evaluate(board), Move()};
    }
    if(maximizing){
        std::pair<int, Move> best = {-INF, Move()};
        for(auto &move:moves){
            board.makeMove(move);
            std::pair<int, Move> result = minimax(board, depth-1, false);
            board.unmakeMove(move);
            if(result.first >= best.first){
                best.first = result.first;
                best.second = move;
            }
        }
        return best;
    }
    else{
        std::pair<int, Move> best = {INF, Move()};
        for(auto &move:moves){
            board.makeMove(move);
            std::pair<int, Move> result = minimax(board, depth-1, true);
            board.unmakeMove(move);
            if(result.first <= best.first){
                best.first = result.first;
                best.second = move;
            }
        }
        return best;
    }
}

int main(){
    Board board("r1b3kr/ppp1Bp1p/1b6/n2P4/2p3q1/2Q2N2/P4PPP/RN2R1K1 w - - 1 0");
    Movelist moves;
    movegen::legalmoves(moves, board);
    bool maximizing = board.sideToMove() == Color::WHITE;
    std::pair<int, Move> result = minimax(board, 5, maximizing);
    std::cout << uci::moveToUci(result.second) << std::endl;
}
