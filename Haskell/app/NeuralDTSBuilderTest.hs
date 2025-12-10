{-# LANGUAGE OverloadedStrings #-}

module NeuralDTSBuilderTest where

import qualified Data.Map.Strict as M
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import qualified Data.Text.Read as TR
import Data.Maybe (mapMaybe)
import Data.List (maximum)
import NeuralDTSBuilderUtils (angleChild, angleParent, sigmoid)

-- word,dim1,dim2,dim3,... 形式でロードする(単語に対して複数の埋め込みを格納)
loadWordEmbeddings :: FilePath -> IO (M.Map T.Text [[Float]])
loadWordEmbeddings path = do
    contents <- TIO.readFile path
    let ls = drop 1 $ T.lines contents

    let parseFloat t =
            case TR.double t of
              Right (d, _) -> Just (realToFrac d :: Float)
              _ -> Nothing

    let parseLine line =
            case T.splitOn "," line of
              (wordTxt : dimsTxt) ->
                  let dims = mapMaybe parseFloat dimsTxt
                  in Just (wordTxt, dims)
              _ -> Nothing

    let parsedLines = mapMaybe parseLine ls
    pure $ M.fromListWith (++) [(word, [dims]) | (word, dims) <- parsedLines]


calcPScore :: Float -> [Float] -> [Float] -> Float
calcPScore alpha pEmb cEmb =
    let a1 = angleChild alpha pEmb cEmb
        a2 = angleParent alpha pEmb cEmb
        score = a1 - a2
        pScore = sigmoid score
    in pScore


neuralDTSBuilder :: IO (T.Text -> T.Text -> Float)
neuralDTSBuilder = do
    let embPath   = "data_dag2all/embeddings_dag2all.csv"
    let alpha     = 0.1
    let threshold = 0.00
    let notFoundValue = 0.0 :: Float

    embMap <- loadWordEmbeddings embPath

    let oracle :: T.Text -> T.Text -> Float
        oracle parent child =
            case (M.lookup parent embMap, M.lookup child embMap) of
                (Just pEmbs, Just cEmbs) ->
                    let allScores = 
                            [ calcPScore alpha pEmb cEmb 
                            | pEmb <- pEmbs
                            , cEmb <- cEmbs
                            ]
                    in if null allScores
                       then notFoundValue
                       else maximum allScores
                _ -> notFoundValue

    pure oracle

main :: IO ()
main = do
    oracle <- neuralDTSBuilder
    print $ oracle "車" "消防車"
    print $ oracle "有袋動物" "カンガルー"
    print $ oracle "鳥類" "ツバメ"
    print $ oracle "果物" "りんご"
    print $ oracle "生き物" "恐竜"
    print $ oracle "車" "カンガルー"  -- 関係のない語彙ペア
    print $ oracle "鳥類" "りんご"
    print $ oracle "哺乳類" "タイ人"
    print $ oracle "調味料" "タルタルソース"
    print $ oracle "調味料" "角砂糖"

