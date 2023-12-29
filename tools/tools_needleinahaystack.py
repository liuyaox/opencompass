import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tiktoken
from matplotlib.colors import LinearSegmentedColormap


class CDMEDatasetProcessor:

    def __init__(self,
                 path,
                 output_path,
                 tokenizer_model='gpt-4',
                 num_records_per_file=10,
                 length_buffer=200,
                 guided=False,
                 file_list=[]):
        self.path = path
        self.output_path = output_path
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        self.num_records_per_file = num_records_per_file
        self.length_buffer = length_buffer
        self.guided = guided
        self.file_list = file_list

    def process_files(self,
                      context_lengths,
                      needle,
                      retrieval_question,
                      document_depth_percent_intervals,
                      document_depth_percent_interval_type='linear'):
        files = Path(self.path).glob('*.jsonl')
        for file in files:
            if os.path.basename(file) in self.file_list:
                self.process_file(file, context_lengths, needle,
                                  retrieval_question,
                                  document_depth_percent_intervals,
                                  document_depth_percent_interval_type)

    def process_file(self, file, context_lengths, needle, retrieval_question,
                     document_depth_percent_intervals,
                     document_depth_percent_interval_type):
        with open(file, 'r', encoding='utf-8') as f:
            lines = [json.loads(line.strip()) for line in f]

        # set output file
        output_file = Path(self.output_path) / f'{file.stem}_processed.jsonl'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as out_f:
            for original_context_length in context_lengths:
                context_length = original_context_length - self.length_buffer
                target_length_per_record = context_length - len(
                    self._get_tokens_from_context(needle))

                for depth_percent in self._generate_depth_percents(
                        document_depth_percent_intervals,
                        document_depth_percent_interval_type):

                    counter = 0
                    accumulated_tokens = []
                    for line in lines:
                        tokens_current_line = self._get_tokens_from_context(
                            line['text'])
                        accumulated_tokens.extend(tokens_current_line)

                        if len(accumulated_tokens) >= target_length_per_record:
                            processed_text = self._generate_context(
                                accumulated_tokens[:target_length_per_record],
                                depth_percent, needle)

                            processed_prompt = self._generate_prompt(
                                processed_text, retrieval_question)
                            json.dump(
                                {
                                    'prompt': processed_prompt,
                                    'answer': needle,
                                    'length': original_context_length,
                                    'depth': int(depth_percent),
                                },
                                out_f,
                                ensure_ascii=False)
                            out_f.write('\n')
                            counter += 1
                            if counter >= self.num_records_per_file:
                                break
                            # reset accumulated_tokens for next record
                            accumulated_tokens = []

    def _generate_context(self, tokens_context, depth_percent, needle):
        tokens_needle = self._get_tokens_from_context(needle)

        # Insert the needle into the context at the specified depth percent
        insertion_point = int(len(tokens_context) * (depth_percent / 100))
        tokens_context = (tokens_context[:insertion_point] + tokens_needle +
                          tokens_context[insertion_point:])

        # Decode the tokens back to text
        new_context = self._decode_tokens(tokens_context)
        return new_context

    def _get_tokens_from_context(self, context):
        return self.tokenizer.encode(context)

    def _decode_tokens(self, tokens):
        return self.tokenizer.decode(tokens)

    def _generate_prompt(self, context, retrieval_question):
        if self.guided:
            prompt = ('你是一个善于回答用户问题的智能AI助手\n'
                      '请保持你的回答简洁清楚。不要说和下面文档中的无关的话，或重复你的回答\n'
                      f'用户现在给你的文档是{context}\n\n'
                      f'现在请问：{retrieval_question}'
                      f'提示：文档中与该问题最相关的句子是_______')
        else:
            prompt = ('你是一个善于回答用户问题的智能AI助手\n'
                      '请保持你的回答简洁清楚。不要说和下面文档中的无关的话，或重复你的回答\n'
                      f'用户现在给你的文档是{context}\n\n'
                      f'现在请问：{retrieval_question}')
        return prompt

    def _generate_depth_percents(self, intervals, interval_type):
        if interval_type == 'linear':
            return np.linspace(0, 100, num=intervals)
        elif interval_type == 'sigmoid':
            return [self._logistic(x) for x in np.linspace(0, 100, intervals)]
        else:
            raise ValueError('Unsupported interval type')

    @staticmethod
    def _logistic(x, L=100, x0=50, k=0.1):
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)


class CDMEDataset():

    @staticmethod
    def generate(processed_datasets_path, data_path, tokenizer_model,
                 num_records_per_file, length_buffer, guided, file_list,
                 context_lengths, needle, retrieval_question,
                 document_depth_percent_intervals):
        # Check if the processed datasets directory exists
        if os.path.exists(processed_datasets_path):
            shutil.rmtree(processed_datasets_path)
            print('The existing processed datasets directory '
                  f'{processed_datasets_path} has been '
                  'removed for a fresh start.')
        else:
            print('No existing processed datasets directory found at'
                  f' {processed_datasets_path}. '
                  'Starting with a fresh directory.')

        processor = CDMEDatasetProcessor(
            path=data_path,
            output_path=processed_datasets_path,
            tokenizer_model=tokenizer_model,
            num_records_per_file=num_records_per_file,
            length_buffer=length_buffer,
            guided=guided,
            file_list=file_list)

        processor.process_files(context_lengths, needle, retrieval_question,
                                document_depth_percent_intervals)

        print('Datasets has been created.')

    @staticmethod
    def visualize(csv_file_paths):
        for file_path in csv_file_paths:
            df = pd.read_csv(file_path)

            # Split 'dataset' column to
            # get 'Context Length' and 'Document Depth'
            df['Context Length'] = df['dataset'].apply(
                lambda x: int(x.split('Length')[1].split('Depth')[0]))
            df['Document Depth'] = df['dataset'].apply(
                lambda x: float(x.split('Depth')[1].split('_')[0]))

            # Exclude 'Context Length' and 'Document Depth' columns
            model_columns = [
                col for col in df.columns
                if col not in ['Context Length', 'Document Depth']
            ]

            for model_name in model_columns[4:]:
                model_df = df[['Document Depth', 'Context Length',
                               model_name]].copy()
                model_df.rename(columns={model_name: 'Score'}, inplace=True)

                # Create pivot table
                pivot_table = pd.pivot_table(model_df,
                                             values='Score',
                                             index=['Document Depth'],
                                             columns=['Context Length'],
                                             aggfunc='mean')

                # Calculate mean scores
                mean_scores = pivot_table.mean().values

                # Calculate overall score
                overall_score = mean_scores.mean()

                # Create heatmap and line plot
                plt.figure(figsize=(17.5, 8))
                ax = plt.gca()
                cmap = LinearSegmentedColormap.from_list(
                    'custom_cmap', ['#F0496E', '#EBB839', '#0CD79F'])

                # Draw heatmap
                sns.heatmap(pivot_table,
                            cmap=cmap,
                            ax=ax,
                            cbar_kws={'label': 'Score'},
                            vmin=0,
                            vmax=100)

                # Set line plot data
                x_data = [i + 0.5 for i in range(len(mean_scores))]
                y_data = mean_scores

                # Create twin axis for line plot
                ax2 = ax.twinx()
                # Draw line plot
                ax2.plot(x_data,
                         y_data,
                         color='white',
                         marker='o',
                         linestyle='-',
                         linewidth=2,
                         markersize=8,
                         label='Average Depth Score')
                # Set y-axis range
                ax2.set_ylim(0, 100)

                # Hide original y-axis ticks and labels
                ax2.set_yticklabels([])
                ax2.set_yticks([])

                # Add legend
                ax2.legend(loc='upper left')

                # Set chart title and labels
                ax.set_title(f'{model_name} 8K Context Performance\n' +
                             'Fact Retrieval Across Context Lengths ' +
                             '("Needle In A Haystack")')
                ax.set_xlabel('Token Limit')
                ax.set_ylabel('Depth Percent')
                ax.set_xticklabels(pivot_table.columns.values, rotation=45)
                ax.set_yticklabels(pivot_table.index.values, rotation=0)
                # Add overall score as a subtitle
                plt.text(0.5,
                         -0.13, f'Overall Score for {model_name}: '
                         f'{overall_score:.2f}',
                         ha='center',
                         va='center',
                         transform=ax.transAxes,
                         fontsize=13)

                # Save heatmap as PNG
                png_file_path = file_path.replace('.csv', f'_{model_name}.png')
                # plt.tight_layout()
                plt.savefig(png_file_path)
                plt.show()

                plt.close()  # Close figure to prevent memory leaks

                # Print saved PNG file path
                print(f'Heatmap for {model_name} saved as: {png_file_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate CDMEDataset.')

    parser.add_argument('--processed_datasets_path',
                        type=str,
                        default='./data/CDME/processed')
    parser.add_argument('--data_path', type=str, default='./data/CDME')
    parser.add_argument('--tokenizer_model', type=str, default='gpt-4')
    parser.add_argument('--num_records_per_file', type=int, default=10)
    parser.add_argument('--length_buffer', type=int, default=200)
    parser.add_argument('--guided', type=bool, default=False)
    parser.add_argument('--file_list', nargs='*', default=['zh_finance.jsonl'])
    parser.add_argument('--context_lengths',
                        nargs='*',
                        type=int,
                        default=list(range(1000, 9000, 1000)))
    parser.add_argument('--needle',
                        type=str,
                        default='\n小明最喜欢的实习的地点就是上海人工智能实验室。\n')
    parser.add_argument('--retrieval_question',
                        type=str,
                        default='小明最喜欢的实习地点是哪里？'
                        '你的回答格式应该为“小明最喜欢的实习地点就是________。”')
    parser.add_argument('--document_depth_percent_intervals',
                        type=int,
                        default=35)
    parser.add_argument('--plot',
                        action='store_true',
                        help='Visualize the dataset results')
    parser.add_argument('--csv_file_paths',
                        nargs='*',
                        default=['path/to/your/result.csv'],
                        help='Paths to CSV files for visualization')

    args = parser.parse_args()

    if args.plot:
        if not args.csv_file_paths:
            print("Error: '--csv_file_paths' is required for visualization.")
            exit(1)
        CDMEDataset.visualize(args.csv_file_paths)

    else:
        doc_depth_intervals = args.document_depth_percent_intervals
        CDMEDataset.generate(
            processed_datasets_path=args.processed_datasets_path,
            data_path=args.data_path,
            tokenizer_model=args.tokenizer_model,
            num_records_per_file=args.num_records_per_file,
            length_buffer=args.length_buffer,
            guided=args.guided,
            file_list=args.file_list,
            context_lengths=args.context_lengths,
            needle=args.needle,
            retrieval_question=args.retrieval_question,
            document_depth_percent_intervals=doc_depth_intervals)


if __name__ == '__main__':
    main()
