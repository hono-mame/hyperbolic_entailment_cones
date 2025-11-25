{-# LANGUAGE OverloadedStrings #-}

module NeuralDTSBuilderNoPickle where

import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Read as TR
import System.IO
import Data.Maybe (fromMaybe, mapMaybe)
import Data.List (foldl')
import NeuralDTSBuilderUtils (angleChild, angleParent)

loadVocab :: FilePath -> IO (M.Map String Int)
loadVocab path = do
    contents <- TIO.readFile path
    let ls = T.lines contents
    let parseLine line =
            case T.splitOn "\t" line of
              [idxTxt, wordTxt] ->
                  case TR.decimal idxTxt of
                    Right (i, _) -> Just (T.unpack wordTxt, i)
                    _ -> Nothing
              _ -> Nothing
    pure $ M.fromList (mapMaybe parseLine ls)


loadEmbeddings :: FilePath -> IO (M.Map Int [Float])
loadEmbeddings path = do
    contents <- TIO.readFile path
    let ls = drop 1 $ T.lines contents  -- 1行目はheader

    let parseFloat t =
            case TR.double t of
              Right (d, _) -> Just (realToFrac d :: Float)
              _ -> Nothing

    let parseLine line =
            case T.splitOn "," line of
              (idxTxt : dimsTxt) ->
                  case TR.decimal idxTxt of
                    Right (idx, _) ->
                        let dims = mapMaybe parseFloat dimsTxt
                        in Just (idx, dims)
                    _ -> Nothing
              _ -> Nothing

    pure $ M.fromList (mapMaybe parseLine ls)


neuralDTSBuilder :: IO (String -> String -> Float)
neuralDTSBuilder = do
    let embPath   = "data/task-0percent_dim-5_class-HypCones_init_class-PoincareNIPS_neg_sampl_strategy-true_neg_lr-0.0003_epochs-300_opt-rsgd_where_not_to_sample-children_neg_edges_attach-parent_lr_init-0.03_epochs_init-100_n_word_vectors.csv"
    let vocabPath = "data/jp_nouns_head_10000_closure.tsv.vocab"
    let alpha     = 0.1
    let threshold = 0.5

    vocabMap <- loadVocab vocabPath
    embMap   <- loadEmbeddings embPath
    let notFoundValue = -1.0 :: Float

    let oracle :: String -> String -> Float
        oracle parent child =
            let mpIdx = M.lookup parent vocabMap
                mcIdx = M.lookup child  vocabMap
            in case (mpIdx, mcIdx) of
                (Just pIdx, Just cIdx) ->
                    case (M.lookup pIdx embMap, M.lookup cIdx embMap) of
                        (Just pEmb, Just cEmb) ->
                            let a1 = angleChild alpha pEmb cEmb
                                a2 = angleParent alpha pEmb cEmb
                                score = a1 - a2
                            -- in if score > threshold then 1.0 else 0.0
                            in score
                        _ -> notFoundValue
                _ -> notFoundValue

    pure oracle

main :: IO ()
main = do
    oracle <- neuralDTSBuilder
    print $ oracle "有袋動物" "経済"
    print $ oracle "有袋動物" "カンガルー"
    print $ oracle "有袋動物" "aaa" -- vocabにない単語なので-1.0が返るはず

