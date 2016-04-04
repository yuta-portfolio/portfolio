using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MsTest
{
    class Breadth
    {
        int width, height;
        int[] puzzle;
        public Breadth(int _width, int _height,int[]_puzzle) {
            width = _width;
            height = _height;
            puzzle = _puzzle;
        }

        public void Solve() { 
        //探索時間を計測するストップウォッチ
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            //解答のステップ数
            int step=0; 
            //これから訪問する状態を入れておくオープンリスト
            Queue<State> openList = new Queue<State>();
            ISet<int> openLists = new HashSet<int>();
            //既に訪問済みの状態を入れておくクローズドリスト
            ISet<int> closedList = new HashSet<int>();

            //変数の初期設定
            int count = 0;
            int[,] firstNums = new int[height,width];
            int firstZeroX = -1, firstZeroY = -1;

            //パズルを作る
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    firstNums[y,x] = puzzle[count++];
                    if (firstNums[y,x] == 0)
                    {
                        firstZeroY = y;
                        firstZeroX = x;
                    }
                }
            }


            /*           A*アルゴリズム                                */
            //初期状態オブジェクトを生成してセットとキューに入れる
            State firstState = new State(null, -1, firstNums, firstZeroY, firstZeroX,0);
            closedList.Add(firstState.hashCode());
            //出発状態をopenListに入れる
            openList.Enqueue(firstState);
            openLists.Add(firstState.hashCode());
            while (openList.Count > 0)//openListが空なら探索は失敗で実行を終了する
            {
                step++;
                //openListの先頭データを取り出してnowStateに入れる
                State nowState = openList.Dequeue();

                //nowStateが目標状態状態なら移動順を列挙して終了
                //逆から辿っているのでひっくり返して出力
                if (nowState.hashCode() == 123456780)
                {
                    //ストップウォッチを停止
                    sw.Stop();
                    LinkedList<String> moves = new LinkedList<String>();
                    LinkedList<State> movess = new LinkedList<State>();
                    while (nowState.preMove != -1)
                    {
                        moves.AddFirst(State.vName[nowState.preMove]);
                        movess.AddFirst(nowState);
                        nowState = nowState.preState;
                    }
                    foreach (String move in moves)
                    {
                        System.Console.Out.WriteLine("move" + move);
                    }
                    //パズルを表示する
                    foreach (State move in movess)
                    {
                        move.showState();
                    }

                    Console.WriteLine(sw.ElapsedMilliseconds + "ms " + step + "step");
                    System.Console.Out.WriteLine("done");
                    Console.ReadLine();
                    return;
                }

                /*ここから改造*/
                //nowStateから移動できる状態がまだ調べてなければ
                //closedListとopenListに追加する
                foreach (State nextState in nowState.nextMove())
                {

                    //ここからA*改造
                    /*
                      nextStateに対してコストを計する
                      nextStateのハッシュがopenに含まれてなければopenに
                     * 含まれており、
                     * closeから削除しopenに入れる
                     */

                    if (!closedList.Contains(nextState.hashCode()))
                    {
                        closedList.Add(nextState.hashCode());
                        openList.Enqueue(nextState);
                    }

                    //closedListに含まれなければclosedListにハッシュコードを入れ、openListに追加
                    //if (!closedList.Contains(nextState.hashCode()) && !openLists.Contains(nextState.hashCode()))
                    //{
                    //    openList.Enqueue(nextState);
                    //    openLists.Add(nextState.hashCode());
                    //}
                   // else if (closedList.Contains(nextState.hashCode()) || openLists.Contains(nextState.hashCode()))
                   // {

                   // }
                }
            }

            //解が存在しなかった時の処理
            Console.WriteLine("no answer");
            Console.ReadLine();
            return;

        }
    }
}

